"""Official mini-SWE-agent control loop adapted to local SGLang serving.

The agent state machine, prompts, structured bash protocol, observation format,
format-error recovery, limits, and submission contract come from the pinned
``mini-swe-agent`` package.  Only the model and environment transports are
adapters: model calls use this repository's SGLang router, while shell actions
use the already instrumented per-instance Docker sandbox.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import yaml
from jinja2 import StrictUndefined, Template

from agentic_kv_request import (
    add_agentic_kv_metadata,
    build_agentic_extra_key,
    confirm_agentic_generation_final,
    confirm_agentic_generation_tool,
    lifecycle_enabled,
)
from pd_metrics import sglang_meta_attrs
from slime.dashboard.api import span as dashboard_span
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .harness import _create_task, _verifier_semaphore
from .verifier import (
    capture_repository_patch,
    patch_metadata,
    prepare_repository_baseline,
    run_inline_verifier,
)


_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_EXECUTORS: dict[int, concurrent.futures.ThreadPoolExecutor] = {}
_EXECUTOR_LOCK = threading.Lock()


def _agent_executor(workers: int) -> concurrent.futures.ThreadPoolExecutor:
    if workers < 1:
        raise ValueError("mini-SWE-agent worker count must be positive")
    with _EXECUTOR_LOCK:
        executor = _EXECUTORS.get(workers)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="pd-mini-swe",
            )
            _EXECUTORS[workers] = executor
        return executor


@lru_cache(maxsize=1)
def _official_config() -> tuple[dict[str, Any], str, str]:
    """Read the benchmark config from the imported, pinned upstream package."""

    try:
        import minisweagent
    except ImportError as exc:
        raise RuntimeError(
            "mini-SWE-agent is unavailable; run scripts/tools/prepare_miniswe_agent.sh "
            "and add its src directory to PYTHONPATH"
        ) from exc
    path = minisweagent.package_dir / "config" / "benchmarks" / "swebench.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config, str(minisweagent.__version__), str(path)


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in message.items()
        if key in {"role", "content", "reasoning_content", "tool_calls", "tool_call_id"}
        and value is not None
    }


def _parse_tool_calls(reply: str, turn: int) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse Qwen's native XML tool calls into mini-SWE-agent/OpenAI records."""

    tool_calls: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for index, raw in enumerate(_TOOL_CALL.findall(reply)):
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("name") != "bash":
            raise ValueError("every tool call must name the bash tool")
        arguments = value.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or not isinstance(arguments.get("command"), str):
            raise ValueError("bash tool call is missing its string command argument")
        call_id = f"call_{turn}_{index}"
        encoded_arguments = json.dumps(arguments, ensure_ascii=False)
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": encoded_arguments},
            }
        )
        actions.append({"command": arguments["command"], "tool_call_id": call_id})
    think_match = _THINK.search(reply)
    reasoning = think_match.group(1).strip() if think_match else ""
    visible = _THINK.sub("", reply)
    content = _TOOL_CALL.sub("", visible).replace("<|im_end|>", "").strip()
    return content, reasoning, tool_calls, actions


class SGLangMiniModel:
    """mini-SWE-agent Model protocol implemented by the local SGLang router."""

    def __init__(
        self,
        *,
        args: Any,
        sample: Sample,
        tokenizer: Any,
        metadata: dict[str, Any],
        options: dict[str, Any],
        sampling_params: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
        official_model_config: dict[str, Any],
    ) -> None:
        self.args = args
        self.sample = sample
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.options = options
        self.sampling_params = dict(sampling_params)
        self.loop = loop
        self.official_model_config = dict(official_model_config)
        self.max_context = int(getattr(args, "max_seq_len", None) or 40960)
        self.max_tokens = int(options.get("max_tokens_per_turn", 8192))
        self.enable_thinking = bool(options.get("enable_thinking", True))
        self.format_error_template = str(official_model_config["format_error_template"])
        self.observation_template = str(official_model_config["observation_template"])
        self.turn_metrics: list[dict[str, Any]] = []
        self.output_token_ids: list[int] = []
        self.raw_replies: list[str] = []
        self.model_seconds = 0.0
        self.generation = 0
        self.last_generation: int | None = None

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {
            **self.official_model_config,
            "model_name": str(getattr(self.args, "hf_checkpoint", "local-sglang")),
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "model_type": f"{type(self).__module__}.{type(self).__name__}",
                    "model": {
                        "transport": "sglang-/generate",
                        "temperature": self.sampling_params.get("temperature"),
                        "top_p": self.sampling_params.get("top_p"),
                        "top_k": self.sampling_params.get("top_k"),
                        "max_tokens_per_turn": self.max_tokens,
                        "max_context_tokens": self.max_context,
                        "enable_thinking": self.enable_thinking,
                    },
                }
            }
        }

    def format_message(self, **kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    def _format_error(self, error: str, finish_reason: str, reply: str, metric: dict[str, Any]) -> Exception:
        from minisweagent.exceptions import FormatError

        content = Template(self.format_error_template, undefined=StrictUndefined).render(
            error=error,
            actions=[],
            has_tool_calls=bool(_TOOL_CALL.search(reply)),
            finish_reason=finish_reason,
        )
        return FormatError(
            {
                "role": "user",
                "content": content,
                "extra": {
                    "interrupt_type": "FormatError",
                    "cost": 0.0,
                    "response": {"raw_reply": reply, "metrics": metric},
                },
            }
        )

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        turn = self.generation
        self.generation += 1
        prepared = [_clean_message(message) for message in messages if message.get("role") != "exit"]
        rendered = self.tokenizer.apply_chat_template(
            prepared,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Execute a bash command",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The bash command to execute",
                                }
                            },
                            "required": ["command"],
                        },
                    },
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        # Some Transformers versions return a BatchEncoding from
        # apply_chat_template(tokenize=True).  Encoding the rendered template
        # explicitly keeps the wire payload a plain list[int].
        input_ids = list(self.tokenizer.encode(rendered, add_special_tokens=False))
        remaining = self.max_context - len(input_ids) - 1
        if remaining <= 0:
            from minisweagent.exceptions import LimitsExceeded

            reason = (
                f"mini-SWE-agent history reached the {self.max_context}-token context limit"
            )
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": reason,
                    "extra": {
                        "exit_status": "ContextWindowExceeded",
                        "submission": "",
                    },
                }
            )
        params = dict(self.sampling_params)
        params["max_new_tokens"] = min(self.max_tokens, remaining)
        request_id = None
        if lifecycle_enabled():
            params, request_id = add_agentic_kv_metadata(
                params,
                trajectory_metadata=self.metadata,
                generation=turn,
                tokenizer=self.tokenizer,
                tool_type="shell",
                tool_suffix_markers=("</tool_call>",),
                terminal_markers=("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",),
            )
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": params,
            "return_logprob": False,
        }
        if request_id is not None:
            payload["extra_key"] = build_agentic_extra_key(request_id, params)
        future = asyncio.run_coroutine_threadsafe(
            self._post(turn, payload, params["max_new_tokens"]), self.loop
        )
        output, generation_seconds = future.result(
            timeout=float(getattr(self.args, "sglang_router_request_timeout_secs", 3600)) + 60
        )
        meta = output["meta_info"]
        output_ids = list(output.get("output_ids") or [])
        reply = self.tokenizer.decode(output_ids, skip_special_tokens=False).strip()
        finish_reason = str((meta.get("finish_reason") or {}).get("type") or "stop")
        metric = {
            "turn": turn + 1,
            "input_tokens": len(input_ids),
            "output_tokens": len(output_ids),
            "generation_seconds": generation_seconds,
            "finish_type": finish_reason,
            "cached_tokens": int(meta.get("cached_tokens", 0) or 0),
            "prompt_tokens": int(meta.get("prompt_tokens", len(input_ids)) or 0),
            "model_reply": reply,
        }
        self.turn_metrics.append(metric)
        self.output_token_ids.extend(output_ids)
        self.raw_replies.append(reply)
        self.model_seconds += generation_seconds
        self.last_generation = turn
        confirm_agentic_generation_tool(
            self.metadata,
            turn,
            p_ready_dir=str(getattr(self.args, "pd_p_ready_dir", "") or ""),
        )
        if finish_reason == "abort":
            raise RuntimeError("SGLang aborted the mini-SWE-agent model request")
        try:
            content, reasoning, tool_calls, actions = _parse_tool_calls(reply, turn)
        except (json.JSONDecodeError, ValueError) as exc:
            raise self._format_error(str(exc), finish_reason, reply, metric) from exc
        if not tool_calls:
            raise self._format_error(
                "No tool calls found in the response. Every response MUST include at least one tool call.",
                finish_reason,
                reply,
                metric,
            )
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": tool_calls,
            "extra": {
                "actions": actions,
                "cost": 0.0,
                "timestamp": time.time(),
                "response": {
                    "finish_reason": "tool_calls",
                    "raw_reply": reply,
                    "metrics": metric,
                },
            },
        }
        if reasoning:
            message["reasoning_content"] = reasoning
        return message

    async def _post(
        self, turn: int, payload: dict[str, Any], max_new_tokens: int
    ) -> tuple[dict[str, Any], float]:
        url = f"http://{self.args.sglang_router_ip}:{self.args.sglang_router_port}/generate"
        with dashboard_span(
            self.args,
            self.sample,
            "generation_turn",
            attrs={
                "task_type": self.metadata.get("task_type", "swe_bench"),
                "turn": turn + 1,
                "max_new_tokens": max_new_tokens,
                "agent_harness": "mini-swe-agent",
                "route_mode": "colocated",
            },
        ) as span:
            started = time.monotonic()
            output = await post(url, payload)
            seconds = time.monotonic() - started
            span.update(sglang_meta_attrs(output.get("meta_info", {})))
        return output, seconds

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from minisweagent.models.utils.actions_toolcall import (
            format_toolcall_observation_messages,
        )

        result = format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            observation_template=self.observation_template,
            template_vars=template_vars,
        )
        if self.turn_metrics:
            observation = "\n".join(str(item.get("content") or "") for item in result)
            self.turn_metrics[-1]["observation"] = observation
            self.turn_metrics[-1]["observation_chars"] = len(observation)
            self.turn_metrics[-1]["observation_tokens"] = len(
                self.tokenizer.encode(observation, add_special_tokens=False)
            )
            actions = message.get("extra", {}).get("actions", [])
            self.turn_metrics[-1]["action"] = "shell"
            self.turn_metrics[-1]["commands"] = [item.get("command", "") for item in actions]
        return result


class MiniDockerEnvironment:
    """mini-SWE-agent Environment protocol over the async instrumented sandbox."""

    def __init__(
        self,
        task: Any,
        loop: asyncio.AbstractEventLoop,
        official_environment_config: dict[str, Any],
    ) -> None:
        self.task = task
        self.loop = loop
        self.environment_config = dict(official_environment_config)
        self.config = SimpleNamespace(**self.environment_config)
        self.shell_calls = 0
        self.tool_seconds = 0.0

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {**self.environment_config, **kwargs}

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment_type": f"{type(self).__module__}.{type(self).__name__}",
                    "environment": self.environment_config,
                }
            }
        }

    def execute(self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        del cwd
        from minisweagent.exceptions import Submitted

        command = str(action.get("command") or "")
        started = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(
            self.task.execute(
                command,
                timeout=timeout or int(self.environment_config.get("timeout", 60)),
                phase="agent_tool",
                environment={
                    str(key): str(value)
                    for key, value in self.environment_config.get("env", {}).items()
                },
                interpreter=tuple(self.environment_config.get("interpreter", ("bash", "-c"))),
            ),
            self.loop,
        )
        try:
            returncode, output = future.result(
                timeout=float(timeout or self.environment_config.get("timeout", 60)) + 15
            )
            exception_info = ""
        except Exception as exc:
            returncode, output = -1, ""
            exception_info = f"An error occurred while executing the command: {exc}"
        self.shell_calls += 1
        self.tool_seconds += time.monotonic() - started
        result = {
            "output": output,
            "returncode": returncode,
            "exception_info": exception_info,
        }
        lines = output.lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and returncode == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": submission,
                    },
                }
            )
        return result


async def _capture_workspace_patch_best_effort(
    task: Any,
    preserved_commit: str,
    metadata: dict[str, Any],
) -> str:
    """Capture diagnostic workspace state without changing an agent outcome."""

    if not preserved_commit:
        return ""
    try:
        return await capture_repository_patch(task, preserved_commit)
    except Exception as exc:
        metadata["workspace_patch_capture_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return ""


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    metadata = dict(sample.metadata or {})
    sample.metadata = metadata
    dataset_id = str(metadata.get("dataset_id") or metadata.get("task_type") or "swe_bench")
    options = dict(getattr(args, "workload_dataset_options", {}).get(dataset_id, {}))
    config, mini_version, config_path = _official_config()
    state = GenerateState(args)
    task = _create_task(metadata, options)
    loop = asyncio.get_running_loop()
    started_at = time.monotonic()
    preserved_commit = ""
    final_patch = ""
    workspace_patch = ""
    reward: int | None = None
    status = Sample.Status.COMPLETED
    stop_reason = "unknown"
    agent = None
    model = None
    environment = None
    verifier_mode = str(options.get("verifier_mode", "inline")).strip().lower()
    try:
        from minisweagent.agents.default import DefaultAgent

        await task.start()
        baseline = await prepare_repository_baseline(task, str(metadata.get("base_commit", "")))
        preserved_commit = baseline.image_commit
        metadata["repository_baseline"] = {
            "official_base_commit": baseline.official_base_commit,
            "official_tree": baseline.official_tree,
            "image_commit": baseline.image_commit,
            "image_tree": baseline.image_tree,
            "kind": baseline.kind,
            "image_commits_ahead": baseline.image_commits_ahead,
            "fingerprint": baseline.fingerprint,
            "source_image_commit": baseline.source_image_commit,
            "initial_worktree_status": baseline.initial_worktree_status,
        }
        model = SGLangMiniModel(
            args=args,
            sample=sample,
            tokenizer=state.tokenizer,
            metadata=metadata,
            options=options,
            sampling_params=sampling_params,
            loop=loop,
            official_model_config=config["model"],
        )
        environment = MiniDockerEnvironment(task, loop, config["environment"])
        agent_config = dict(config["agent"])
        agent_config["cost_limit"] = 0.0
        agent_config["step_limit"] = int(options.get("step_limit", agent_config["step_limit"]))
        agent_config["wall_time_limit_seconds"] = int(options.get("wall_time_limit_seconds", 3600))
        agent_config["max_consecutive_format_errors"] = int(
            options.get("max_consecutive_format_errors", 3)
        )
        agent = DefaultAgent(model, environment, **agent_config)
        executor = _agent_executor(int(options.get("agent_workers", getattr(args, "sglang_server_concurrency", 128))))
        result = await loop.run_in_executor(
            executor,
            agent.run,
            str(metadata["problem_statement"]),
        )
        exit_status = str(result.get("exit_status") or "")
        final_patch = str(result.get("submission") or "")
        stop_reason = exit_status or "agent_exit"
        if exit_status != "Submitted":
            status = Sample.Status.TRUNCATED
        if exit_status == "Submitted":
            workspace_patch = await _capture_workspace_patch_best_effort(
                task, preserved_commit, metadata
            )
        else:
            metadata["workspace_patch_capture_skipped"] = "not_submitted"
        if model.last_generation is not None:
            confirm_agentic_generation_final(
                metadata,
                model.last_generation,
                p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
            )
        metadata.update(patch_metadata(final_patch))
        metadata["model_patch"] = final_patch
        metadata["workspace_patch_chars"] = len(workspace_patch)
        metadata["workspace_patch_sha256"] = hashlib.sha256(
            workspace_patch.encode("utf-8")
        ).hexdigest()
        metadata["submission_matches_workspace_patch"] = final_patch == workspace_patch
        if verifier_mode == "inline" and exit_status == "Submitted":
            verifier_limit = int(options.get("verifier_max_concurrent", 16))
            queued = time.monotonic()
            async with _verifier_semaphore(args, verifier_limit):
                metadata["verifier_queue_seconds"] = time.monotonic() - queued
                verifier = await run_inline_verifier(
                    task,
                    metadata,
                    final_patch,
                    timeout_seconds=float(options.get("verifier_timeout_seconds", 2400)),
                    output_tail_chars=int(options.get("verifier_output_tail_chars", 12000)),
                )
            metadata["swe_bench_verifier"] = verifier.to_metadata()
            reward = verifier.reward
            if verifier.status == "infrastructure_error":
                status = Sample.Status.FAILED
                stop_reason = "verifier_infrastructure_error"
        elif verifier_mode == "inline":
            # SWE-bench only grades an explicitly submitted patch.  Running
            # hidden tests for a wall/step/context-limited trajectory adds no
            # information and can turn a valid zero-score outcome into a
            # spurious infrastructure failure.
            reward = 0
            metadata["swe_bench_verifier"] = {
                "status": "not_submitted",
                "resolved": False,
                "reward": 0,
                "report": None,
                "test_exit_code": None,
                "timed_out": False,
                "duration_seconds": 0.0,
                "output_tail": "",
                "error_type": None,
                "error": None,
            }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        status = Sample.Status.FAILED
        stop_reason = f"environment_error:{type(exc).__name__}"
        metadata["environment_error"] = str(exc)
    finally:
        await task.close()

    trajectory = agent.save(None) if agent is not None else {}
    turn_metrics = model.turn_metrics if model is not None else []
    sample.status = status
    sample.tokens = list(model.output_token_ids) if model is not None else []
    sample.response = "\n\n".join(model.raw_replies) if model is not None else ""
    sample.response_length = len(sample.tokens)
    sample.reward = reward
    sample.tool_time = environment.tool_seconds if environment is not None else 0.0
    sample.tool_call_count = environment.shell_calls if environment is not None else 0
    sample.code_call_count = 0
    sample.search_call_count = 0
    metadata.update(
        {
            "num_turns": len(turn_metrics),
            "stop_reason": stop_reason,
            "shell_call_count": sample.tool_call_count,
            "tool_time": sample.tool_time,
            "model_time": model.model_seconds if model is not None else 0.0,
            "turn_metrics": turn_metrics,
            "sample_time": time.monotonic() - started_at,
            "container_image": task.image,
            "sandbox_metrics": task.metrics,
            "sandbox_backend": str(options.get("sandbox_backend", "docker")).lower(),
            "verifier_mode": verifier_mode,
            "mini_swe_agent_version": mini_version,
            "mini_swe_agent_config_path": config_path,
            "mini_swe_agent_trajectory_format": trajectory.get("trajectory_format"),
            "mini_swe_agent_trajectory": trajectory,
            "workspace_patch_chars": len(workspace_patch),
        }
    )
    return sample
