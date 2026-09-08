"""Multi-turn SWE-bench inference harness backed by isolated Docker tasks."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_kv_request import (
    add_agentic_kv_metadata,
    build_agentic_extra_key,
    confirm_agentic_generation_final,
    confirm_agentic_generation_tool,
    generation_has_visible_content,
    lifecycle_enabled,
)
from pd_metrics import sglang_meta_attrs
from slime.dashboard.api import span as dashboard_span
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from .verifier import (
    capture_repository_patch,
    daytona_api_key,
    patch_metadata,
    prepare_repository_baseline,
    run_inline_verifier,
)


_BASH_BLOCK = re.compile(r"```(?:bash|sh)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_DONE_MARKER = "SWE_TASK_COMPLETE"


def command_from_reply(reply: str) -> str | None:
    """Extract exactly one fenced shell action; never execute free-form prose."""

    visible = _THINK_BLOCK.sub("", reply.replace("<|im_end|>", "")).strip()
    matches = _BASH_BLOCK.findall(visible)
    if len(matches) != 1:
        return None
    command = matches[0].strip()
    return command or None


def _truncate_output(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    keep = max(1, (limit - 96) // 2)
    removed = len(value) - 2 * keep
    return f"{value[:keep]}\n...[truncated {removed} characters]...\n{value[-keep:]}"


def _image_name(metadata: dict[str, Any], options: dict[str, Any]) -> str:
    if metadata.get("image_name"):
        return str(metadata["image_name"])
    instance = str(metadata["instance_id"]).lower().replace("__", "_1776_")
    template = str(
        options.get(
            "image_template",
            "docker.io/swebench/sweb.eval.x86_64.{instance}:latest",
        )
    )
    return template.format(instance=instance, instance_id=metadata["instance_id"])


@dataclass
class DockerTask:
    image: str
    command_timeout: float
    start_timeout: float
    network: str
    container_id: str | None = None
    metrics: dict[str, Any] = field(
        default_factory=lambda: {
            "backend": "docker",
            "exec_calls": [],
            "uploads": [],
        }
    )

    async def _run_host(self, *args: str, timeout: float) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise
        return process.returncode or 0, output.decode("utf-8", errors="replace")

    async def start(self) -> None:
        inspect_started = time.monotonic()
        inspect_code, _ = await self._run_host(
            "docker", "image", "inspect", self.image, timeout=60
        )
        self.metrics["image_inspect_seconds"] = time.monotonic() - inspect_started
        self.metrics["image_present_before_start"] = inspect_code == 0
        name = f"pd-swe-{uuid.uuid4().hex[:12]}"
        command = [
            "docker", "run", "-d", "--rm", "--name", name,
        ]
        run_id = os.environ.get("PD_SWE_RUN_ID", "").strip()
        if run_id:
            command.extend(["--label", f"pd.swe.run_id={run_id}"])
        command.extend([
            "--entrypoint", "sleep", "--network", self.network,
            "-w", "/testbed", self.image, "86400",
        ])
        start_started = time.monotonic()
        returncode, output = await self._run_host(*command, timeout=self.start_timeout)
        self.metrics["container_start_seconds"] = time.monotonic() - start_started
        if returncode:
            raise RuntimeError(f"docker run failed for {self.image}: {output.strip()}")
        self.container_id = output.strip()
        if not self.container_id:
            raise RuntimeError(f"docker run returned no container id for {self.image}")
        # Official images can use a non-root repository owner. Mark only the
        # isolated /testbed checkout safe, without modifying the host config.
        await self.execute(
            "git config --global --add safe.directory /testbed",
            timeout=30,
            phase="sandbox_setup",
        )

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        phase: str = "agent_tool",
        environment: dict[str, str] | None = None,
        interpreter: tuple[str, ...] = ("bash", "-lc"),
    ) -> tuple[int, str]:
        if not self.container_id:
            raise RuntimeError("SWE-bench container is not running")
        seconds = max(1, int(timeout or self.command_timeout))
        started = time.monotonic()
        docker_environment: list[str] = []
        for key, value in (environment or {}).items():
            docker_environment.extend(("-e", f"{key}={value}"))
        returncode, output = await self._run_host(
            "docker",
            "exec",
            "-w",
            "/testbed",
            *docker_environment,
            self.container_id,
            "timeout",
            "--signal=KILL",
            f"{seconds}s",
            *interpreter,
            command,
            timeout=seconds + 10,
        )
        self.metrics["exec_calls"].append(
            {
                "phase": phase,
                "duration_seconds": time.monotonic() - started,
                "timeout_seconds": seconds,
                "exit_code": returncode,
                "command_chars": len(command),
                "output_chars": len(output),
            }
        )
        return returncode, output

    async def upload_bytes(self, contents: bytes, path: str) -> None:
        if not self.container_id:
            raise RuntimeError("SWE-bench container is not running")
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile(prefix="pd-swe-upload-", delete=False) as handle:
                handle.write(contents)
                temporary_path = handle.name
            started = time.monotonic()
            returncode, output = await self._run_host(
                "docker",
                "cp",
                temporary_path,
                f"{self.container_id}:{path}",
                timeout=self.start_timeout,
            )
            if returncode:
                raise RuntimeError(
                    f"docker cp failed for {self.container_id}:{path}: {output.strip()}"
                )
            self.metrics["uploads"].append(
                {
                    "path": path,
                    "bytes": len(contents),
                    "duration_seconds": time.monotonic() - started,
                }
            )
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

    async def close(self) -> None:
        container_id, self.container_id = self.container_id, None
        if not container_id:
            return
        started = time.monotonic()
        with contextlib.suppress(Exception):
            await asyncio.shield(
                self._run_host("docker", "rm", "-f", container_id, timeout=60)
            )
        self.metrics["container_close_seconds"] = time.monotonic() - started


def _daytona_response_text(response: Any) -> str:
    for name in ("result", "output", "stdout"):
        value = getattr(response, name, None)
        if value is not None:
            return str(value)
    return str(response)


def _daytona_exit_code(response: Any) -> int:
    for name in ("exit_code", "exitCode", "code"):
        value = getattr(response, name, None)
        if value is not None:
            return int(value)
    return 0


@dataclass
class DaytonaTask:
    """Optional remote backend matching ``DockerTask``'s small sandbox API.

    Daytona is imported only when selected, so local Docker evaluation has no
    cloud dependency.  The interface follows the Daytona SDK used by Miles.
    """

    image: str
    command_timeout: float
    start_timeout: float
    api_key: str
    instance_id: str
    cpu: int = 4
    memory_gb: int = 16
    disk_gb: int = 30
    keep_sandbox: bool = False
    sandbox: Any = None
    client: Any = None
    metrics: dict[str, Any] = field(
        default_factory=lambda: {
            "backend": "daytona",
            "exec_calls": [],
            "uploads": [],
        }
    )

    async def start(self) -> None:
        try:
            from daytona import (
                CreateSandboxFromImageParams,
                Daytona,
                DaytonaConfig,
                Resources,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Daytona backend requires the optional daytona SDK"
            ) from exc
        self.client = Daytona(DaytonaConfig(api_key=self.api_key))
        params = CreateSandboxFromImageParams(
            image=self.image,
            os_user="root",
            env_vars={
                "OMP_NUM_THREADS": str(self.cpu),
                "OPENBLAS_NUM_THREADS": str(self.cpu),
                "MKL_NUM_THREADS": str(self.cpu),
                "PYTHONUNBUFFERED": "1",
            },
            resources=Resources(
                cpu=self.cpu,
                memory=self.memory_gb,
                disk=self.disk_gb,
            ),
            labels={"pd-swebench-instance-id": self.instance_id},
            auto_stop_interval=0,
            auto_delete_interval=0,
        )
        started = time.monotonic()
        self.sandbox = await asyncio.to_thread(
            self.client.create, params, timeout=self.start_timeout
        )
        self.metrics["container_start_seconds"] = time.monotonic() - started

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        phase: str = "agent_tool",
    ) -> tuple[int, str]:
        if self.sandbox is None:
            raise RuntimeError("Daytona SWE-bench sandbox is not running")
        seconds = max(1, int(timeout or self.command_timeout))
        started = time.monotonic()
        try:
            response = await asyncio.to_thread(
                self.sandbox.process.exec,
                f"cd /testbed && {command}",
                timeout=seconds,
            )
        except Exception as exc:
            is_process_timeout = type(exc).__name__ == (
                "DaytonaProcessExecutionTimeoutError"
            ) or (
                getattr(exc, "source", None) == "DAYTONA_DAEMON"
                and getattr(exc, "code", None) == "PROCESS_EXECUTION_TIMEOUT"
            )
            if is_process_timeout:
                raise TimeoutError(str(exc)) from exc
            raise
        exit_code = _daytona_exit_code(response)
        output = _daytona_response_text(response)
        self.metrics["exec_calls"].append(
            {
                "phase": phase,
                "duration_seconds": time.monotonic() - started,
                "timeout_seconds": seconds,
                "exit_code": exit_code,
                "command_chars": len(command),
                "output_chars": len(output),
            }
        )
        return exit_code, output

    async def upload_bytes(self, contents: bytes, path: str) -> None:
        if self.sandbox is None:
            raise RuntimeError("Daytona SWE-bench sandbox is not running")
        started = time.monotonic()
        await asyncio.to_thread(self.sandbox.fs.upload_file, contents, path)
        self.metrics["uploads"].append(
            {
                "path": path,
                "bytes": len(contents),
                "duration_seconds": time.monotonic() - started,
            }
        )

    async def close(self) -> None:
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is None or self.keep_sandbox:
            return
        started = time.monotonic()
        await asyncio.shield(
            asyncio.to_thread(self.client.delete, sandbox, timeout=180)
        )
        self.metrics["container_close_seconds"] = time.monotonic() - started


def _create_task(
    metadata: dict[str, Any], options: dict[str, Any]
) -> DockerTask | DaytonaTask:
    backend = str(options.get("sandbox_backend", "docker")).strip().lower()
    common = {
        "image": _image_name(metadata, options),
        "command_timeout": float(options.get("command_timeout_seconds", 300)),
        "start_timeout": float(options.get("container_start_timeout_seconds", 600)),
    }
    if backend == "docker":
        return DockerTask(
            **common,
            network=str(options.get("container_network", "none")),
        )
    if backend == "daytona":
        return DaytonaTask(
            **common,
            api_key=daytona_api_key(str(options.get("daytona_api_key", ""))),
            instance_id=str(metadata["instance_id"]),
            cpu=int(options.get("daytona_cpu", 4)),
            memory_gb=int(options.get("daytona_memory_gb", 16)),
            disk_gb=int(options.get("daytona_disk_gb", 30)),
            keep_sandbox=bool(options.get("daytona_keep_sandbox", False)),
        )
    raise ValueError(f"unsupported SWE-bench sandbox_backend {backend!r}")


def _initial_tokens(tokenizer: Any, metadata: dict[str, Any], options: dict[str, Any]) -> list[int]:
    system = str(
        options.get(
            "system_prompt",
            "You are a software engineering agent working in /testbed. "
            "Inspect and modify the repository to solve the issue. On every turn, return exactly "
            "one shell command in one ```bash fenced block; explanatory text may appear outside "
            "the block. Commands run in a fresh shell but repository changes persist. Do not edit "
            "tests. When the implementation is complete, return ```bash\\necho SWE_TASK_COMPLETE\\n```.",
        )
    )
    user = (
        f"SWE-bench instance: {metadata['instance_id']}\n\n"
        f"<problem_statement>\n{metadata['problem_statement']}\n</problem_statement>\n\n"
        "Work directly in /testbed. Begin by inspecting the relevant files."
    )
    kwargs: dict[str, Any] = {}
    if "enable_thinking" in options:
        kwargs["enable_thinking"] = bool(options["enable_thinking"])
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def _observation_tokens(tokenizer: Any, output: str) -> list[int]:
    rendered = (
        "\n<|im_end|>\n<|im_start|>user\n"
        f"<tool_result>\n{output}\n</tool_result>\n"
        "Continue with exactly one fenced shell command, or echo SWE_TASK_COMPLETE when done."
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def _verifier_semaphore(args: Any, limit: int) -> asyncio.Semaphore:
    if limit < 1:
        raise ValueError("verifier_max_concurrent must be positive")
    semaphores = getattr(args, "_swe_bench_verifier_semaphores", None)
    if semaphores is None:
        semaphores = {}
        setattr(args, "_swe_bench_verifier_semaphores", semaphores)
    semaphore = semaphores.get(limit)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        semaphores[limit] = semaphore
    return semaphore


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    metadata = dict(sample.metadata or {})
    sample.metadata = metadata
    dataset_id = str(metadata.get("dataset_id") or metadata.get("task_type") or "swe_bench")
    options = dict(getattr(args, "workload_dataset_options", {}).get(dataset_id, {}))
    state = GenerateState(args)
    prompt_tokens = _initial_tokens(state.tokenizer, metadata, options)
    response_tokens: list[int] = []
    max_context = int(getattr(args, "max_seq_len", None) or 40960)
    max_turns = int(options.get("max_turns", 32))
    per_turn_cap = int(options.get("max_tokens_per_turn", 4096))
    output_limit = int(options.get("max_observation_chars", 12000))
    verifier_mode = str(options.get("verifier_mode", "disabled")).strip().lower()
    if verifier_mode not in {"disabled", "capture", "inline"}:
        raise ValueError(
            "SWE-bench verifier_mode must be disabled, capture, or inline; "
            f"got {verifier_mode!r}"
        )
    task = _create_task(metadata, options)
    started_at = time.monotonic()
    tool_seconds = 0.0
    model_seconds = 0.0
    command_count = 0
    turn_metrics: list[dict[str, Any]] = []
    last_generation: int | None = None
    status = Sample.Status.COMPLETED
    stop_reason = "max_turns"
    final_patch = ""
    preserved_commit = ""
    reward: int | None = None
    try:
        await task.start()
        if verifier_mode in {"capture", "inline"}:
            baseline = await prepare_repository_baseline(
                task, str(metadata.get("base_commit", ""))
            )
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
        else:
            head_exit, head_output = await task.execute(
                "git rev-parse --verify HEAD", timeout=60, phase="baseline"
            )
            if head_exit == 0:
                preserved_commit = head_output.strip().lower()
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        for turn in range(max_turns):
            input_ids = prompt_tokens + response_tokens
            remaining = max_context - len(input_ids) - 1
            if remaining <= 0:
                status, stop_reason = Sample.Status.TRUNCATED, "budget"
                break
            params = dict(sampling_params)
            params["max_new_tokens"] = min(per_turn_cap, remaining)
            request_id = None
            if lifecycle_enabled():
                params, request_id = add_agentic_kv_metadata(
                    params,
                    trajectory_metadata=metadata,
                    generation=turn,
                    tokenizer=state.tokenizer,
                    tool_type="shell",
                    tool_suffix_markers=("```",),
                    terminal_markers=(_DONE_MARKER,),
                )
            payload: dict[str, Any] = {
                "input_ids": input_ids,
                "sampling_params": params,
                "return_logprob": False,
            }
            if request_id is not None:
                payload["extra_key"] = build_agentic_extra_key(request_id, params)
            with dashboard_span(
                args,
                sample,
                "generation_turn",
                attrs={
                    "task_type": metadata.get("task_type", "swe_bench"),
                    "turn": turn + 1,
                    "max_new_tokens": params["max_new_tokens"],
                    "route_mode": "strict_pd",
                },
            ) as span:
                generation_started = time.monotonic()
                output = await post(url, payload)
                generation_seconds = time.monotonic() - generation_started
                model_seconds += generation_seconds
                span.update(sglang_meta_attrs(output.get("meta_info", {})))
            last_generation = turn
            meta = output["meta_info"]
            new_tokens = list(output.get("output_ids") or [])
            response_tokens.extend(new_tokens)
            reply = state.tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
            finish_type = meta["finish_reason"]["type"]
            turn_metric: dict[str, Any] = {
                "turn": turn + 1,
                "input_tokens": len(input_ids),
                "output_tokens": len(new_tokens),
                "generation_seconds": generation_seconds,
                "finish_type": finish_type,
                "cached_tokens": int(meta.get("cached_tokens", 0) or 0),
                "prompt_tokens": int(meta.get("prompt_tokens", len(input_ids)) or 0),
                # Keep a structured, human-readable trajectory in addition to
                # the flattened token stream stored on the Sample.
                "model_reply": reply,
            }
            turn_metrics.append(turn_metric)
            if finish_type == "abort":
                status, stop_reason = Sample.Status.ABORTED, "abort"
                break
            if not generation_has_visible_content(new_tokens, state.tokenizer):
                status, stop_reason = Sample.Status.TRUNCATED, "empty_generation"
                break
            command = command_from_reply(reply)
            if command is None:
                turn_metric["action"] = "invalid"
                confirm_agentic_generation_tool(
                    metadata, turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                observation = "Invalid response: return exactly one command in one ```bash block."
            elif _DONE_MARKER in command:
                turn_metric["action"] = "complete"
                confirm_agentic_generation_final(
                    metadata, turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                stop_reason = "task_complete"
                break
            else:
                confirm_agentic_generation_tool(
                    metadata, turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                tool_started = time.monotonic()
                returncode, command_output = await task.execute(
                    command, phase="agent_tool"
                )
                command_seconds = time.monotonic() - tool_started
                tool_seconds += command_seconds
                command_count += 1
                turn_metric.update(
                    {
                        "action": "shell",
                        "command": command,
                        "command_chars": len(command),
                        "command_seconds": command_seconds,
                        "command_exit_code": returncode,
                        "command_output_chars": len(command_output),
                    }
                )
                observation = _truncate_output(
                    f"<returncode>{returncode}</returncode>\n<output>\n{command_output}\n</output>",
                    output_limit,
                )
            observation_ids = _observation_tokens(state.tokenizer, observation)
            remaining = max_context - len(prompt_tokens) - len(response_tokens) - 1
            if len(observation_ids) > remaining:
                observation_ids = observation_ids[: max(0, remaining)]
                status, stop_reason = Sample.Status.TRUNCATED, "budget"
            response_tokens.extend(observation_ids)
            turn_metric["observation_tokens"] = len(observation_ids)
            turn_metric["observation"] = state.tokenizer.decode(
                observation_ids, skip_special_tokens=False
            )
            if status is Sample.Status.TRUNCATED:
                break
            if finish_type == "length":
                status, stop_reason = Sample.Status.TRUNCATED, "length"
                break
        else:
            status = Sample.Status.TRUNCATED
        if last_generation is not None and stop_reason != "task_complete":
            confirm_agentic_generation_final(
                metadata, last_generation,
                p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
            )
        if preserved_commit:
            final_patch = await capture_repository_patch(task, preserved_commit)
        else:
            patch_exit, final_patch = await task.execute(
                "git diff --no-ext-diff --binary",
                timeout=120,
                phase="patch_capture",
            )
            if patch_exit != 0:
                raise RuntimeError(
                    f"final repository patch capture failed with code {patch_exit}"
                )
        metadata.update(patch_metadata(final_patch))
        if bool(options.get("store_model_patch", verifier_mode != "disabled")):
            metadata["model_patch"] = final_patch

        if verifier_mode == "inline":
            verifier_limit = int(options.get("verifier_max_concurrent", 4))
            verifier_queued_at = time.monotonic()
            async with _verifier_semaphore(args, verifier_limit):
                metadata["verifier_queue_seconds"] = (
                    time.monotonic() - verifier_queued_at
                )
                verifier_result = await run_inline_verifier(
                    task,
                    metadata,
                    final_patch,
                    timeout_seconds=float(options.get("verifier_timeout_seconds", 2400)),
                    output_tail_chars=int(options.get("verifier_output_tail_chars", 12000)),
                )
            metadata["swe_bench_verifier"] = verifier_result.to_metadata()
            reward = verifier_result.reward
            if verifier_result.status == "infrastructure_error":
                status = Sample.Status.FAILED
                stop_reason = "verifier_infrastructure_error"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        status = Sample.Status.FAILED
        stop_reason = f"environment_error:{type(exc).__name__}"
        metadata["environment_error"] = str(exc)
    finally:
        await task.close()

    sample.status = status
    sample.tokens = prompt_tokens + response_tokens
    sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    sample.reward = reward
    sample.tool_time = tool_seconds
    sample.tool_call_count = command_count
    sample.code_call_count = 0
    sample.search_call_count = 0
    metadata.update(
        {
            "num_turns": (last_generation + 1) if last_generation is not None else 0,
            "stop_reason": stop_reason,
            "shell_call_count": command_count,
            "tool_time": tool_seconds,
            "model_time": model_seconds,
            "turn_metrics": turn_metrics,
            "sample_time": time.monotonic() - started_at,
            "container_image": task.image,
            "sandbox_metrics": task.metrics,
            "sandbox_backend": str(options.get("sandbox_backend", "docker")).lower(),
            "verifier_mode": verifier_mode,
            **patch_metadata(final_patch),
        }
    )
    return sample
