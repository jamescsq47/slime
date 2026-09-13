"""BFCL official simulated tools/checker, local model; bounded engineering probe."""
import argparse
import ast
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import requests
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.dabstep.probe import ContainerPython, action_text
from data.scienceagentbench.probe import write_json


def decode_calls(text, allowed):
    """Accept only a list of public tool calls with literal arguments, never code."""
    text = action_text(text, []).strip()
    if text.startswith('```'):
        lines = text.splitlines()
        text = '\n'.join(lines[1:-1]).strip() if lines[-1].strip() == '```' else text
    if not text.startswith('['):
        return []  # Official prompting convention: non-call answer ends user turn.
    expr = ast.parse(text, mode='eval').body
    if not isinstance(expr, ast.List):
        raise ValueError('Expected function-call list')
    calls = []
    for call in expr.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id not in allowed:
            raise ValueError('Unknown or unsafe tool call')
        # Official resolve_ast_by_type converts bare names to strings (e.g.
        # true -> 'true', not True). Match that without its unsafe eval branches.
        class NamesToStrings(ast.NodeTransformer):
            def visit_Name(self, node):
                return ast.copy_location(ast.Constant(node.id), node)
        for i, arg in enumerate(call.args):
            call.args[i] = NamesToStrings().visit(arg)
            ast.literal_eval(call.args[i])
        for kw in call.keywords:
            if kw.arg is None:
                raise ValueError('Expanded kwargs are not allowed')
            kw.value = NamesToStrings().visit(kw.value)
            ast.literal_eval(kw.value)
        calls.append(ast.unparse(call))
    return calls


def common_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def tools_and_prompt(row, root):
    sys.path.insert(0, str(root))
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING
    from bfcl_eval.constants.default_prompts import (
        PROMPT_STYLE_TEMPLATES, PROMPT_TEMPLATE_MAPPING, OUTPUT_FORMAT_MAPPING)
    tools = []
    for cls in row['involved_classes']:
        path = root / 'bfcl_eval/data/multi_turn_func_doc' / MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls]
        tools.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    # Match upstream populate_test_cases_with_predefined_functions: Base exposes
    # all involved-class tools; excluded_function is not used by that loader.
    style = PROMPT_STYLE_TEMPLATES['classic']
    prompt = PROMPT_TEMPLATE_MAPPING['plaintext'].format(
        persona=style['persona'], task=style['task'],
        tool_call_format=style['tool_call_no_tag'].format(output_format=OUTPUT_FORMAT_MAPPING['python'], param_types=''),
        multiturn_behavior=style['multiturn_behavior'],
        available_tools=style['available_tools'].format(format='json', functions=json.dumps(tools, indent=4)))
    return {t['name'] for t in tools}, prompt


def init_executor(executor, row):
    # No ground truth enters the inference sandbox until all model calls stop.
    row = {k: row[k] for k in ('id', 'initial_config', 'involved_classes')}
    executor(f'''
import sys, json, time
sys.path.insert(0, '/opt/bfcl')
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
row = json.loads({json.dumps(row)!r})
execute_multi_turn_func_call([], row['initial_config'], row['involved_classes'], 'live', row['id'])
final_answer('ready')
''')


def execute_call(executor, call):
    return json.loads(executor(f'''
started = time.perf_counter()
responses, _ = execute_multi_turn_func_call([{call!r}], row['initial_config'], row['involved_classes'], 'live', row['id'])
elapsed = time.perf_counter() - started
final_answer(json.dumps({{'response': responses[0], 'seconds': elapsed}}))
''').output)


def score(executor, row, predicted, gold):
    # Official checker replays both model and gold with fresh, separate state.
    return json.loads(executor(f'''
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
result = multi_turn_checker(json.loads({json.dumps(predicted)!r}), json.loads({json.dumps(gold)!r}), row, 'multi_turn_base', 'score')
final_answer(json.dumps(result, default=str))
''').output)


def run_task(row, gold, config, output, tokenizer, model_call=None):
    start = time.monotonic()
    result = dict(id=row['id'], state='running', calls=[], tools=[], prefix_checks=[],
                  question=row['question'], predicted=[[] for _ in row['question']])
    allowed, prompt = tools_and_prompt(row, Path(config['upstream']))
    messages = [{'role': 'system', 'content': prompt}]
    previous_ids = None
    executor = None
    def checkpoint():
        write_json(output / f"progress-{row['id']}.json", {**result, 'elapsed_seconds': time.monotonic()-start})
    try:
        executor = ContainerPython(Path(__file__).parent, 30, image=config['image'])
        init_executor(executor, row)
        for turn, user_messages in enumerate(row['question']):
            messages.extend(copy.deepcopy(user_messages))
            for step in range(config['max_steps_per_turn']):
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
                if len(ids) + config['max_new_tokens'] > config['context_length']:
                    raise ValueError('Context budget exceeded; history is not truncated')
                if previous_ids is not None:
                    matched = common_prefix(previous_ids, ids)
                    result['prefix_checks'].append(dict(turn=turn, step=step, parent_tokens=len(previous_ids),
                        reused_prefix_tokens=matched, exact=matched == len(previous_ids)))
                checkpoint()
                before = time.monotonic()
                if model_call:
                    answer = model_call(ids, messages)
                else:
                    response = requests.post(config['endpoint']+'/generate', json={
                        'input_ids': ids, 'sampling_params': {'temperature': config['temperature'],
                        'top_p': 1, 'top_k': -1, 'max_new_tokens': config['max_new_tokens']}}, timeout=300)
                    response.raise_for_status()
                    answer = response.json()
                result['calls'].append(dict(turn=turn, step=step, messages=copy.deepcopy(messages),
                    input_ids=ids, output=answer, wall_seconds=time.monotonic()-before))
                previous_ids = ids + answer['output_ids']
                messages.append({'role': 'assistant', 'content': answer['text']})
                try:
                    calls = decode_calls(answer['text'], allowed)
                except (ValueError, SyntaxError) as exc:
                    # Official Base handler ends this user turn on decode error,
                    # then continues the remaining scripted user turns.
                    result.setdefault('parse_errors', []).append(dict(turn=turn, step=step, error=str(exc)))
                    result['predicted'][turn].append([])
                    checkpoint()
                    break
                result['predicted'][turn].append(calls)
                checkpoint()
                if not calls:
                    break
                for call in calls:
                    before = time.monotonic()
                    event = execute_call(executor, call)
                    result['tools'].append(dict(turn=turn, step=step, call=call, **event,
                                                rpc_seconds=time.monotonic()-before))
                    messages.append({'role': 'tool', 'name': call, 'content': event['response']})
                checkpoint()
            else:
                result['step_limit_hit'] = True
                break
        result['state'] = 'completed'
        result['score'] = score(executor, row, result['predicted'], gold)
    except Exception as exc:
        result.update(state='error', error=repr(exc))
    finally:
        if executor:
            executor.cleanup()
        result['cleanup_errors'] = executor.cleanup_errors if executor else []
        result['wall_seconds'] = time.monotonic()-start
        checkpoint()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config['upstream'])
    config['upstream_commit'] = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    config['image'] = subprocess.check_output(['docker','image','inspect',config['image'],'--format','{{.Id}}'],text=True).strip()
    data = root/'bfcl_eval/data/BFCL_v4_multi_turn_base.json'
    gold_path = root/'bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json'
    rows = [json.loads(line) for line in data.read_text().splitlines()][:config['num_tasks']]
    gold = {r['id']:r['ground_truth'] for r in map(json.loads,gold_path.read_text().splitlines())}
    config['input_sha256'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (data,gold_path)}
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'config.json',config)
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
        futures = [pool.submit(run_task,r,gold[r['id']],config,args.output,tokenizer) for r in rows]
        for future in as_completed(futures):
            r = future.result()
            write_json(args.output/f"task-{r['id']}.json",r)
            results.append(r)
            summary = dict(tasks=len(results), expected=len(rows), elapsed_seconds=time.monotonic()-started,
                passed=sum(r.get('score',{}).get('valid',False) for r in results),
                errors=sum(r['state']=='error'for r in results), all_terminal=len(results)==len(rows))
            write_json(args.output/'summary.json',summary)
            print(json.dumps({**summary,'last_id':r['id']}),flush=True)


if __name__ == '__main__':
    main()
