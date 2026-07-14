# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import re
import time
from typing import Any

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import compute_score as math_dapo_compute_score
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e

# Import tool sandbox functionality
from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

# Jinja2 template for tool-enabled conversations
TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_conversation_with_tools(
    prompt: str, tools: list[dict[str, Any]] = None, system_prompt: str = None, messages: list[dict[str, Any]] = None
) -> str:
    """Format conversation using Jinja2 template with tool support"""
    template = Template(TOOL_TEMPLATE)

    # Prepare messages
    messages_to_render = []

    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = (
            "You are a helpful assistant that can use Python "
            "tools to solve mathematical problems. When you need "
            "to perform calculations, use the code_interpreter "
            "tool to execute code and get results."
        )

    messages_to_render.append({"role": "system", "content": system_content})

    # Add user message if provided
    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)

    # Render template
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])

    return formatted_text


def postprocess_predictions(prediction: str):
    """Extract action and content from prediction string"""
    # Check for Answer: \boxed{...} format (only format we need for math_dapo)
    # Use a more robust regex that handles nested braces
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content

    # Also accept bare \boxed{...} without requiring "Answer:" prefix
    bare_boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    bare_boxed_match = re.search(bare_boxed_pattern, prediction, re.DOTALL)
    if bare_boxed_match:
        content = bare_boxed_match.group(1).strip()
        return "answer", content

    # Then check for <tool_call> tags (new format from Jinja2 template)
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            import json

            # Clean up the JSON string by removing newlines and extra
            # whitespace
            json_str = tool_call_match.group(1)
            # Replace newlines in string values with \n
            json_str = json_str.replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})

            if tool_name == "code_interpreter":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    # Then check for <code> tags
    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content

    # Finally check for ```python code blocks (lowest priority)
    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    # Handle <tool_call> tags (new format from Jinja2 template)
    if "<tool_call>" in resp:
        # Find the last occurrence of <tool_call>...</tool_call>
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle <code> tags
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle Answer: \boxed{...} format (only format we need for math_dapo)
    if "Answer:" in resp and "\\boxed{" in resp:
        # Find the last occurrence of Answer: \boxed{...} with nested braces support
        answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Also handle bare \boxed{...} without "Answer:" prefix
    if "\\boxed{" in resp:
        bare_boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(bare_boxed_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    return resp


def reconstruct_loss_masks(response: str, tokenizer) -> list:
    """Reconstruct loss masks from response content.
    Used when resuming a partial rollout.
    """
    try:
        response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
        loss_masks = [1] * len(response_tokens)

        interpreter_pattern = r'<interpreter>(.*?)</interpreter>'
        matches = list(re.finditer(interpreter_pattern, response, re.DOTALL))

        if not matches:
            return loss_masks

        for match in matches:
            start_char = match.start()
            end_char = match.end()

            prefix = response[:start_char]
            prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            start_token_idx = len(prefix_tokens)

            prefix_with_interpreter = response[:end_char]
            prefix_with_interp_tokens = tokenizer(prefix_with_interpreter, add_special_tokens=False)["input_ids"]
            end_token_idx = len(prefix_with_interp_tokens)

            for i in range(start_token_idx, end_token_idx):
                if i < len(loss_masks):
                    loss_masks[i] = 0

        return loss_masks

    except Exception as e:
        print(f"[WARNING] Error reconstructing loss masks: {e}")
        response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
        loss_masks = [1] * len(response_tokens)
        return loss_masks


def count_tool_turns(response: str) -> int:
    """Count the number of completed tool turns in the response.
    Used to determine where to resume generation.
    """
    interpreter_count = response.count("</interpreter>")
    return interpreter_count


async def execute_predictions(prediction: str) -> str:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Content is already the Python code (extracted by
        # postprocess_predictions)
        code = content.strip()
        if code:
            async with SEMAPHORE:
                result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found" "\n</interpreter>\n\n"
            done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "If I want to execute code, I should put the code between "
            "<code> and </code>. "
            "If I want to give the final answer, I should use the format "
            "'Answer: \\boxed{answer}'. Let me try again.\n"
        )
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls with partial rollout support"""

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Initialize total_off_policy_tokens if it doesn't exist
    if not hasattr(sample, 'total_off_policy_tokens'):
        sample.total_off_policy_tokens = 0

    # Set up tool specs
    tool_specs = tool_registry.get_tool_specs()

    # Ensure metadata exists
    if not hasattr(sample, 'metadata') or sample.metadata is None:
        sample.metadata = {}

    # Check if this is a partial rollout resume
    if args.partial_rollout and sample.status == Sample.Status.ABORTED and sample.response:
        # Partial rollout: resume from existing response
        metadata = sample.metadata

        if metadata.get("formatted_prompt"):
            prompt = metadata["formatted_prompt"]
        else:
            prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)

        prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # Restore state from saved metadata if available
        response = sample.response
        response_token_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]

        if metadata.get("partial_rollout") and metadata.get("loss_masks") and metadata.get("tool_call_count"):
            loss_masks = metadata["loss_masks"]
            if len(loss_masks) != len(response_token_ids):
                print(f"[WARNING] Saved loss_masks length ({len(loss_masks)}) != response tokens ({len(response_token_ids)})")
                loss_masks = reconstruct_loss_masks(response, state.tokenizer)
            tool_call_count = metadata["tool_call_count"]
            start_turn = metadata.get("current_turn", tool_call_count)
        else:
            loss_masks = reconstruct_loss_masks(response, state.tokenizer)
            tool_call_count = count_tool_turns(response)
            start_turn = tool_call_count

        # Update off-policy token count
        sample.total_off_policy_tokens += len(response_token_ids)
        # Carry over timing from previous attempt(s)
        _accrued_sample_time = getattr(sample, 'sample_time', 0.0) or 0.0
        _tool_time = getattr(sample, 'tool_time', 0.0) or 0.0
    else:
        # Non-partial rollout: start fresh
        sample.rollout_log_probs = None
        sample.response = ""
        sample.response_length = 0
        sample.loss_mask = None

        prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)
        prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response = ""
        response_token_ids = []
        loss_masks = []
        tool_call_count = 0  # Track actual tool call rounds
        start_turn = 0
        _accrued_sample_time = 0.0
        _tool_time = 0.0

    _start_time = time.monotonic()

    if args.rollout_max_context_len is not None:
        max_context_length = args.rollout_max_context_len
    else:
        max_context_length = args.context_parallel_size * args.max_tokens_per_gpu

    output = None

    for turn in range(start_turn, TOOL_CONFIGS["max_turns"]):
        # Check if total length exceeds max context length
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Clamp per-turn max_new_tokens to the remaining context budget so a
        # single turn cannot push total_length past max_context_length. Without
        # this, a turn can append up to rollout_max_response_len tokens on top
        # of a total that was just barely under the cap, producing samples
        # that exceed the training-side max_tokens_per_gpu * cp_size budget
        # and crash the partition/batch code (asserts or OOMs on an oversized
        # partition).
        remaining_budget = max_context_length - total_length
        per_turn_sampling_params = dict(sampling_params)
        per_turn_sampling_params["max_new_tokens"] = min(
            sampling_params.get("max_new_tokens", remaining_budget),
            remaining_budget,
        )

        # Use token IDs instead of text
        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": per_turn_sampling_params,
            "return_logprob": True,  # Request log probabilities for training
        }

        # Log payload to wandb for debugging
        try:
            import wandb

            if wandb.run is not None:
                # Count available tools (from tool_specs)
                available_tools = len(tool_specs)
                # Count tools used in the current response
                tools_used = response.count("<interpreter>")

                wandb.log(
                    {
                        "debug/payload_length": len(prompt + response),
                        "debug/available_tools": available_tools,
                        "debug/tools_used": tools_used,
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass  # wandb not available

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            if not args.partial_rollout:
                sample.status = Sample.Status.ABORTED
                sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
                sample.tool_time = _tool_time
                return sample
            else:
                # Partial rollout enabled: process partial response and save state
                if "output_token_logprobs" in output["meta_info"]:
                    cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                    cur_response = state.tokenizer.decode(cur_response_token_ids)
                    cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
                    if sample.rollout_log_probs is None:
                        sample.rollout_log_probs = []
                    sample.rollout_log_probs += cur_log_probs
                else:
                    sample.status = Sample.Status.ABORTED
                    sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
                    sample.tool_time = _tool_time
                    return sample

                if cur_response:
                    response += cur_response
                    response_token_ids += cur_response_token_ids
                    loss_masks += [1] * len(cur_response_token_ids)

                    _tool_start = time.monotonic()
                    next_obs, done = await execute_predictions(cur_response)
                    if next_obs:
                        if "<interpreter>" in next_obs:
                            tool_call_count += 1
                        _tool_time += time.monotonic() - _tool_start

                        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
                        response += next_obs
                        response_token_ids += obs_tokens_ids
                        loss_masks += [0] * len(obs_tokens_ids)
                        if sample.rollout_log_probs is not None:
                            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

                # Save state for resumption
                sample.status = Sample.Status.ABORTED
                sample.tokens = prompt_tokens_ids + response_token_ids
                sample.response_length = len(response_token_ids)
                sample.response = response
                sample.loss_mask = loss_masks
                sample.tool_call_count = tool_call_count

                sample.payload_text = prompt + response
                sample.payload_has_system = "<|im_start|>system" in prompt + response
                sample.payload_has_tools = "# Tools" in prompt + response

                sample.metadata.update({
                    "partial_rollout": True,
                    "current_turn": turn,
                    "loss_masks": loss_masks,
                    "tool_call_count": tool_call_count,
                    "formatted_prompt": prompt,
                })
                sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
                sample.tool_time = _tool_time
                return sample

        if "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs

        else:
            # sglang returned text but no output_token_logprobs — we cannot
            # recover per-token logprobs for this turn, which would desync
            # rollout_log_probs from response_token_ids and blow up
            # `slice_log_prob_with_cp` downstream. Abort the sample so the
            # fully_async rollout manager returns the whole group to the
            # buffer for retry instead of poisoning the trainer.
            sample.status = Sample.Status.ABORTED
            sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
            sample.tool_time = _tool_time
            return sample

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        _tool_start = time.monotonic()
        next_obs, done = await execute_predictions(cur_response)
        if done:
            break

        # Count tool calls (when we get interpreter output, it means a tool
        # was called)
        if "<interpreter>" in next_obs:
            tool_call_count += 1
            _tool_time += time.monotonic() - _tool_start

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens (they won't be used due to loss_mask=0)
        # Check if maximum tool call count reached
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_tokens_ids)

            assert len(response_token_ids) == len(
                sample.rollout_log_probs
            ), f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(sample.rollout_log_probs)} logps"

        # Tool output is appended verbatim and can push total_length past
        # max_context_length (the per-turn generation was clamped to the
        # remaining budget, but tool output is unconstrained). Trim tail
        # tokens so the final sample fits the training budget exactly.
        overflow = len(prompt_tokens_ids) + len(response_token_ids) - max_context_length
        if overflow > 0:
            response_token_ids = response_token_ids[:-overflow]
            loss_masks = loss_masks[:-overflow]
            if sample.rollout_log_probs is not None:
                sample.rollout_log_probs = sample.rollout_log_probs[:-overflow]
            # Resync the text field from the trimmed token list so
            # reward_func's `sample.prompt + sample.response` matches what
            # the model was actually trained on. decode(tokenize(text)) can
            # be lossy on some tokenizers (whitespace / special-token
            # collapse), but reward_func's regex is whitespace-robust and
            # the trainer sees tokens, not text — so the drift is safe.
            response = state.tokenizer.decode(response_token_ids)
            sample.status = Sample.Status.TRUNCATED
            break

        if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
            break

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    # Store payload information for wandb logging
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count

    # Set status based on finish reason
    if output is not None:
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED
    else:
        sample.status = Sample.Status.TRUNCATED

    sample.sample_time = _accrued_sample_time + (time.monotonic() - _start_time)
    sample.tool_time = _tool_time
    return sample


async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string
    solution_str = sample.prompt + sample.response

    # Get ground truth answer - label is a string, not a dict
    ground_truth = str(sample.label) if sample.label is not None else ""

    # Get tool call count as num_turns
    num_turns = getattr(sample, "tool_call_count", 0)

    # use \\boxed{...} answer
    result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result