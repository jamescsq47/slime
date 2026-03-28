# Unified generation function supporting both code execution and search tools

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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

# Import search functionality
try:
    from qa_em_format import compute_score_em
except ImportError as e:
    raise ImportError("QA EM format module is not installed") from e

# Import tool sandbox functionality
from tool_sandbox import SEMAPHORE as TOOL_SEMAPHORE, TOOL_CONFIGS, tool_registry

# Configuration for unified tool use
UNIFIED_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 6,  # Increased max turns since we have two tools
    "topk": 3,
    "search_concurrency": 256,
    "tool_concurrency": 256,
    "return_logprob": True,  # Set to True to collect log probabilities for training
    
    # ============== Search Backend Configuration ==============
    "search_backend": "local",  # Options: "local", "google", "duckduckgo"
    
    # ============== Local Search Configuration ==============
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",
        "proxy": None,
    },
    
    # ============== Google Search Configuration ==============
    "google": {
        "api_key": "your_api_key_here",
        "snippet_only": True,
        "proxy": None,
    },
    
    # ============== DuckDuckGo Search Configuration ==============
    "duckduckgo": {
        "proxy": None,
    },
    
    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
    "tool_use_reward": 0.1,  # Reward for appropriate tool use
}

# Semaphores for rate limiting
SEARCH_SEMAPHORE = asyncio.Semaphore(UNIFIED_CONFIGS["search_concurrency"])
TOOL_SEMAPHORE = asyncio.Semaphore(UNIFIED_CONFIGS["tool_concurrency"])

# Jinja2 template for tool-enabled conversations with multiple tool types
UNIFIED_TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant that can use tools to solve problems.
{%- endif %}
{%- if tools %}
# Available Tools

You have access to the following tools to help you solve problems:

<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

## How to Use Tools:

1. **Code Interpreter**: Use this to execute Python code for calculations or data processing.
   Format:
   <tool_call>
   {"name": "code_interpreter", "arguments": {"code": "your Python code here"}}
   </tool_call>

2. **Search**: Use this to search for information online.
   Format:
   <tool_call>
   {"name": "search", "arguments": {"query": "your search query here"}}
   </tool_call>

After receiving tool results, you can either:
- Call another tool if needed
- Provide the final answer using: Answer: \\boxed{your answer}
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


def format_conversation_with_unified_tools(
    prompt: str,
    tools: List[Dict[str, Any]] = None,
    system_prompt: str = None,
    messages: List[Dict[str, Any]] = None,
) -> str:
    """Format conversation using Jinja2 template with unified tool support"""
    template = Template(UNIFIED_TOOL_TEMPLATE)

    # Prepare messages
    messages_to_render = []

    # System prompt
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = (
            "You are a helpful assistant that can use both code execution "
            "and search tools to solve problems. Use code_interpreter for "
            "calculations and data processing. Use search for finding "
            "information online. Always provide your final answer in the "
            "format: Answer: \\boxed{your answer}"
        )

    messages_to_render.append({"role": "system", "content": system_content})

    # Add user message
    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})

    # Add assistant responses from previous turns
    if messages:
        messages_to_render.extend(messages)

    # Render template
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])

    return formatted_text


def get_unified_tool_specs() -> List[Dict[str, Any]]:
    """Get specifications for all available tools"""
    tools = []
    
    # Add code interpreter tool
    tools.append({
        "name": "code_interpreter",
        "description": "Execute Python code for calculations and data processing",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute"
                }
            },
            "required": ["code"]
        }
    })
    
    # Add search tool
    tools.append({
        "name": "search",
        "description": "Search for information online",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    })
    
    return tools


def postprocess_predictions_unified(prediction: str) -> Tuple[Optional[str], str]:
    """Extract action and content from prediction string for unified tools"""
    
    # Check for final answer format
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return "answer", content

    answer_tag_pattern = r"<answer>\s*(.*?)\s*</answer>"
    answer_tag_match = re.search(answer_tag_pattern, prediction, re.DOTALL)
    if answer_tag_match:
        content = answer_tag_match.group(1).strip()
        return "answer", content

    # Check for tool calls
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_match = re.search(tool_call_pattern, prediction, re.DOTALL)
    if tool_call_match:
        try:
            # Clean up JSON string
            json_str = tool_call_match.group(1)
            json_str = json_str.replace("\n", "\\n")
            tool_call_data = json.loads(json_str)
            tool_name = tool_call_data.get("name")
            arguments = tool_call_data.get("arguments", {})

            if tool_name == "code_interpreter":
                code = arguments.get("code", "")
                if code.strip():
                    return "code", code
            elif tool_name == "search":
                query = arguments.get("query", "")
                if query.strip():
                    return "search", query
        except (json.JSONDecodeError, KeyError, AttributeError):
            pass

    # Backward compatibility: check for old formats
    # <code> tags
    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, prediction, re.DOTALL)
    if code_match:
        content = code_match.group(1).strip()
        return "code", content

    # <search> tags
    search_pattern = r"<search>(.*?)</search>"
    search_match = re.search(search_pattern, prediction, re.DOTALL)
    if search_match:
        content = search_match.group(1).strip()
        return "search", content

    # Python code blocks
    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, prediction, re.DOTALL)
    if python_code_match:
        content = python_code_match.group(1).strip()
        return "code", content

    return None, ""


def postprocess_responses_unified(resp: str) -> str:
    """Post-process response to ensure tag completeness"""
    
    # Handle tool calls
    if "<tool_call>" in resp:
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle answer format
    if "Answer:" in resp and "\\boxed{" in resp:
        answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"

    # Backward compatibility
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"
    
    if "</search>" in resp:
        return resp.split("</search>")[0] + "</search>"
    
    if "```python" in resp:
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    
    return resp


async def search_unified(query: str) -> str:
    """Perform search using configured backend"""
    backend = UNIFIED_CONFIGS["search_backend"]

    if backend == "local":
        from local_search_server import local_search
        local_config = UNIFIED_CONFIGS["local"]
        result = await local_search(
            local_config["search_url"],
            query,
            UNIFIED_CONFIGS["topk"],
            proxy=local_config["proxy"],
        )
    elif backend == "google":
        from google_search_server import google_search
        google_config = UNIFIED_CONFIGS["google"]
        result = await google_search(
            google_config["api_key"],
            query,
            UNIFIED_CONFIGS["topk"],
            snippet_only=google_config["snippet_only"],
            proxy=google_config["proxy"],
        )
    elif backend == "duckduckgo":
        from duckduckgo_search_server import duckduckgo_search
        duckduckgo_config = UNIFIED_CONFIGS["duckduckgo"]
        result = await duckduckgo_search(
            query,
            UNIFIED_CONFIGS["topk"],
            proxy=duckduckgo_config.get("proxy"),
        )
    else:
        raise ValueError(f"Unknown search backend: {backend}")

    # Format results
    format_reference = ""
    for idx, doc_item in enumerate(result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

    return format_reference


async def execute_predictions_unified(prediction: str) -> Tuple[str, bool]:
    """Execute predictions and return results for unified tools"""
    action, content = postprocess_predictions_unified(prediction)

    if action == "code":
        code = content.strip()
        if code:
            async with TOOL_SEMAPHORE:
                result = await tool_registry.execute_tool("code_interpreter", {"code": code})
            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found\n</interpreter>\n\n"
            done = False
    
    elif action == "search":
        search_query = content
        async with SEARCH_SEMAPHORE:
            search_results = await search_unified(search_query)
        next_obs = f"\n\n<information>\n{search_results.strip()}\n</information>\n\n"
        done = False
    
    elif action == "answer":
        next_obs = ""
        done = True
    
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "To use a tool, use the format:\n"
            "<tool_call>\n"
            '{"name": "tool_name", "arguments": {"arg": "value"}}\n'
            "</tool_call>\n\n"
            "Available tools: code_interpreter (for Python code) and search (for online information).\n"
            "To provide the final answer, use: Answer: \\boxed{your answer}\n"
        )
        done = False

    return next_obs, done


async def generate_unified(args, sample: Sample, sampling_params) -> Sample:
    """Unified generation function supporting both code execution and search tools"""
    assert not args.partial_rollout, "Partial rollout is not supported for this function at the moment."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Get tool specifications
    tool_specs = get_unified_tool_specs()
    
    # Format initial prompt with tools
    prompt = format_conversation_with_unified_tools(
        prompt=sample.prompt,
        tools=tool_specs,
        system_prompt=getattr(sample, "system_prompt", None)
    )

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    rollout_log_probs = [] if UNIFIED_CONFIGS["return_logprob"] else None
    
    # Tracking metrics
    tool_call_count = 0
    search_call_count = 0
    code_call_count = 0
    tool_time = 0.0
    sample_start = time.time()

    for turn in range(UNIFIED_CONFIGS["max_turns"]):
        # Check max context length
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        max_context_length = args.rollout_max_context_len or (args.context_parallel_size * args.max_tokens_per_gpu)
        
        if total_length >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            break

        # Prepare payload
        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": sampling_params,
        }
        
        if UNIFIED_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        # Make generation request
        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        # Process response
        if UNIFIED_CONFIGS["return_logprob"] and "output_token_logprobs" in output["meta_info"]:
            cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_response_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            rollout_log_probs += cur_log_probs
        else:
            cur_response = output["text"]
            if not UNIFIED_CONFIGS["return_logprob"]:
                cur_response = postprocess_responses_unified(cur_response)
            cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # Check if generation stopped due to length
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        # Execute tool if needed
        start_time = time.time()
        next_obs, done = await execute_predictions_unified(cur_response)
        elapsed_time = time.time() - start_time
        tool_time += elapsed_time

        if done:
            break

        # Track tool usage
        if "<interpreter>" in next_obs:
            tool_call_count += 1
            code_call_count += 1
        if "<information>" in next_obs:
            tool_call_count += 1
            search_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        
        # Add observation to response
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        # Add dummy log probs for observation tokens
        if UNIFIED_CONFIGS["return_logprob"] and rollout_log_probs is not None:
            rollout_log_probs += [0.0] * len(obs_tokens_ids)
            
            # Verify alignment
            assert len(response_token_ids) == len(rollout_log_probs), \
                f"Token/logp length mismatch at turn {turn}: {len(response_token_ids)} tokens vs {len(rollout_log_probs)} logps"

    # Record timing
    sample_time = time.time() - sample_start

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks
    sample.prompt = prompt
    
    # Tool usage metrics
    sample.tool_call_count = tool_call_count
    sample.search_call_count = search_call_count
    sample.code_call_count = code_call_count
    sample.tool_token_count = loss_masks.count(0)
    sample.tool_time = tool_time
    sample.sample_time = sample_time
    
    # Store log probs if enabled
    if UNIFIED_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs

    # Set status
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


async def reward_func_unified(args, sample: Sample, task_type: str = None, **kwargs) -> Dict[str, Any]:
    """Unified reward function supporting both math and QA tasks"""
    
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")
    
    # Get task_type from sample.metadata if not provided
    if task_type is None:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        task_type = metadata.get("task_type", "math")

    solution_str = sample.prompt + sample.response
    
    if task_type == "math":
        # Math task with code interpreter
        ground_truth = sample.label if sample.label is not None else ""
        num_turns = getattr(sample, "tool_call_count", 0)
        
        # Use math_dapo reward
        result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
        
        # Encourage appropriate tool use
        if result["score"] < 0:
            # Reward for using tools appropriately
            tool_reward = min(num_turns * 0.1, 0.5)  # Cap at 0.5
            result["score"] = min(-0.6, result["score"] + tool_reward)
        
        if result["pred"] is None:
            result["pred"] = ""
            
    elif task_type == "qa":
        # QA task with search
        # Handle different label formats
        if isinstance(sample.label, str):
            formatted_gt = {"target": [sample.label]}
        elif isinstance(sample.label, dict):
            # Label can be either {"ground_truth": ...} or {"target": ...}
            if "ground_truth" in sample.label:
                formatted_gt = sample.label.get("ground_truth", {})
            elif "target" in sample.label:
                # Direct target format
                formatted_gt = {"target": sample.label["target"]}
            else:
                # Fallback: wrap entire dict
                formatted_gt = sample.label
        else:
            formatted_gt = {"target": [str(sample.label)]}
            
        result = compute_score_em(
            solution_str=solution_str,
            ground_truth=formatted_gt,
            format_score=UNIFIED_CONFIGS["format_score"],
        )
        
        # Convert to dictionary format for consistency
        result = {"score": result, "pred": ""}  # EM score doesn't provide prediction
        
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be 'math' or 'qa'")

    return result