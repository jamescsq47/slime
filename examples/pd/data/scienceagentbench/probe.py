"""Small real-task probe: upstream SAB Self-debug, local Qwen, Docker execution.

This is not a PD harness or an official correctness/performance benchmark.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.dabstep.probe import ContainerPython, action_text


def upstream_agent(path):
    """Reuse audited official prompt construction, extraction and solve loop verbatim.

Avoid importing its obsolete cloud engine/litellm and host package installation.
    """
    tree = ast.parse(Path(path).read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.endswith('_PROMPT') for t in node.targets
        ):
            ast.literal_eval(node.value)
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == 'ScienceAgent':
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef)
                         and n.name in {'get_sys_msg', 'write_program', 'solve_task'}]
            nodes.append(node)
    namespace = {'Path': Path, 're': re,
                 'trim_messages': lambda messages, *args, **kwargs: messages}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['ScienceAgent']


class LocalEngine:
    def __init__(self, config):
        self.config = config
        self.llm_engine_name = config['model']
        self.client = OpenAI(base_url=config['endpoint'], api_key='local', timeout=300, max_retries=0)
        self.calls = []
        self.on_update = lambda: None

    def respond(self, messages, **_official_sampling):
        start = time.monotonic()
        event = {'messages': messages}
        try:
            result = self.client.chat.completions.create(
                model=self.config['model'], messages=messages,
                temperature=self.config['temperature'], top_p=1,
                max_tokens=self.config['max_new_tokens'], extra_body={'top_k': -1})
            choice = result.choices[0]
            text = action_text(choice.message.content, [])
            event.update(raw_content=choice.message.content, action_content=text,
                         finish_reason=choice.finish_reason, usage=result.usage.model_dump())
            return text, result.usage.prompt_tokens, result.usage.completion_tokens
        except Exception as exc:
            event['error'] = repr(exc)
            raise
        finally:
            event['wall_seconds'] = time.monotonic() - start
            self.calls.append(event)
            self.on_update()


def write_json(path, data):
    """Publish a complete checkpoint; readers never see a partially written JSON."""
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    os.replace(temporary, path)


def save_artifact(executor, output_fname, output, instance_id):
    source = Path(output_fname)
    if source.is_absolute() or '..' in source.parts or source.parts[0] != 'pred_results':
        raise ValueError('Expected output must be under pred_results')
    destination = output / 'artifacts' / str(instance_id) / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    # docker cp cannot see this node's running-container tmpfs. Read through
    # exec instead, bounded in both size and time; never recreate device/FIFO nodes.
    reader = '''
import os, stat, sys
path = sys.argv[1]
if not stat.S_ISREG(os.lstat(path).st_mode):
    raise ValueError("Artifact must be a regular non-symlink file")
fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
with os.fdopen(fd, "rb") as stream:
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Artifact must be a regular file")
    limit = 64 * 1024 * 1024
    if info.st_size > limit:
        raise ValueError("Artifact exceeds capture limit64MiB")
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Artifact grew beyond capture limit64MiB")
sys.stdout.buffer.write(data)
'''
    copied = subprocess.run(['docker', 'exec', executor.name, '/usr/bin/python', '-I', '-S', '-c', reader,
                             f'/workspace/{source}'], capture_output=True, timeout=30)
    if copied.returncode:
        raise ValueError(copied.stderr.decode(errors='replace')[-2000:])
    destination.write_bytes(copied.stdout)
    digest = hashlib.sha256(copied.stdout).hexdigest()
    return {'path': str(destination), 'bytes': destination.stat().st_size, 'sha256': digest}


def execute_program(executor, code, output_fname, timeout):
    """Fresh Python subprocess per official complete-program execution, inside sandbox."""
    # Only the wrapper executes in the persistent worker. Model code is a child process.
    wrapper = f'''
import json, os, pathlib, subprocess, sys, time
pathlib.Path("pred_results").mkdir(exist_ok=True)
pathlib.Path("pred_program.py").write_text({code!r})
target = pathlib.Path({output_fname!r})
if target.is_file():
    target.unlink()
started = time.perf_counter()
try:
    result = subprocess.run([sys.executable, "pred_program.py"], capture_output=True,
                            timeout={timeout!r})
    record = {{"returncode": result.returncode,
              "stdout": result.stdout.decode(errors="replace")[-16000:],
              "stderr": result.stderr.decode(errors="replace")[-8000:],
              "timed_out": False}}
except subprocess.TimeoutExpired:
    record = {{"returncode": None, "stdout": "", "stderr": "Program execution timed out", "timed_out": True}}
record["execution_seconds"] = time.perf_counter() - started
record["output_exists"] = target.is_file()
record["output_bytes"] = target.stat().st_size if target.is_file() else 0
final_answer(json.dumps(record))
'''
    return json.loads(executor(wrapper).output)


def run_task(row, config, output, base_class):
    # Keep only official model inputs; gold program/plan/eval are never mounted or prompted.
    task = {k: str(row[k]) for k in ('task_inst', 'dataset_folder_tree', 'dataset_preview', 'output_fname')}
    folder = task['dataset_folder_tree'].splitlines()[0][4:].strip('/')
    task['dataset_path'] = '/data/context/' + folder
    result = {'instance_id': int(row['instance_id']), 'task': task, 'state': 'started',
              'started_at_unix': time.time()}
    task_image = config.get('task_images', {}).get(str(result['instance_id']), config['image'])
    result['executor_image'] = task_image
    engine = LocalEngine(config)
    executor = None
    executions = []
    start = time.monotonic()

    def checkpoint():
        write_json(output / f"progress-{result['instance_id']}.json",
                   {**result, 'model_calls': engine.calls, 'executions': executions,
                    'elapsed_seconds': time.monotonic() - start})

    engine.on_update = checkpoint

    class Agent(base_class):
        def __init__(self):
            self.llm_engine = engine
            self.llm_cost = {'input_cost_per_token': 0, 'output_cost_per_token': 0}
            self.context_cutoff = config['context_length']
            self.use_self_debug = True
            self.use_knowledge = False
            self.history = []
            self.sys_msg = ''

        def step(self, out_fname, output_fname):
            code = Path(out_fname).read_text()
            result['state'] = 'executing'
            checkpoint()
            event = execute_program(executor, code, output_fname, config['tool_timeout_seconds'])
            event['code'] = code
            executions.append(event)
            checkpoint()
            if event['returncode'] == 0 and event['output_exists']:
                result['state'] = 'executed_output_exists'
                if config.get('save_artifacts', False):
                    try:
                        result['artifact'] = save_artifact(executor, output_fname, output, result['instance_id'])
                    except Exception as exc:
                        result['artifact_error'] = repr(exc)
                return True, 0.0
            if event['timed_out']:
                # Kill the entire sandbox on timeout; don't leave descendants behind.
                executor.cleanup()
                result['state'] = 'tool_timeout'
                return True, 0.0
            if event['returncode'] < 0:
                # A killed child is not a missing-output error. Do not repeatedly
                # execute a resource-exhausting program during this bounded probe.
                result.update(state='program_signal', signal=-event['returncode'])
                executor.cleanup()
                return True, 0.0
            if event['returncode'] != 0:
                error = event['stderr'] or f"The program exited with code {event['returncode']} and no stderr."
            else:
                error = 'The program does not save its output correctly. Please check if the functions are executed and the output path is correct.'
            messages = [{'role': 'user', 'content': self.sys_msg}, self.history[-1],
                        {'role': 'user', 'content': error}]
            result['state'] = 'generating'
            checkpoint()
            answer, _, _ = engine.respond(messages)
            unchanged = self.write_program(answer, out_fname)
            self.history += [{'role': 'user', 'content': error}, {'role': 'assistant', 'content': answer}]
            if unchanged:
                result['state'] = 'unchanged_program'
            return unchanged, 0.0

    try:
        executor = ContainerPython(config['datasets'], config['tool_timeout_seconds'] + 30,
                                   image=task_image, memory=config.get('sandbox_memory', '4g'),
                                   workspace_size=config.get('sandbox_workspace', '512m'))
        # Public library assets are immutable in the image; keep each task's
        # writable Salem/joblib cache local to its disposable sandbox.
        executor('''
import pathlib
assets = pathlib.Path('/opt/scienceagentbench-home/.salem_cache')
if assets.is_dir():
    cache = pathlib.Path.home() / '.salem_cache'
    cache.mkdir(exist_ok=True)
    for asset in assets.glob('salem-sample-data-*'):
        (cache / asset.name).symlink_to(asset, target_is_directory=asset.is_dir())
final_answer('ready')
''')
        agent = Agent()
        result['state'] = 'generating'
        checkpoint()
        result['trajectory'] = agent.solve_task(task, str(output / f"program-{result['instance_id']}.py"))
        if result['state'] in {'started', 'generating', 'executing'}:
            result['state'] = 'max_debug_steps'
    except Exception as exc:
        result.update(state='error', error=repr(exc))
    finally:
        if executor:
            executor.cleanup()
        result.update(model_calls=engine.calls, executions=executions,
                      cleanup_errors=executor.cleanup_errors if executor else [],
                      wall_seconds=time.monotonic() - start, finished_at_unix=time.time())
        checkpoint()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config['image'] = subprocess.check_output(
        ['docker', 'image', 'inspect', config['image'], '--format', '{{.Id}}'], text=True).strip()
    config['task_images'] = {task: subprocess.check_output(
        ['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip()
        for task, image in config.get('task_images', {}).items()}
    rows = pd.read_parquet(config['annotations'])
    if config['task_ids'] == 'all':
        config['task_ids'] = [int(x) for x in rows.instance_id]
    rows = rows[rows.instance_id.isin(config['task_ids'])].to_dict('records')
    assert [int(row['instance_id']) for row in rows] == config['task_ids']
    args.output.mkdir(parents=True, exist_ok=True)
    config['upstream_commit'] = subprocess.check_output(
        ['git', '-C', str(Path(config['upstream_agent']).parent), 'rev-parse', 'HEAD'], text=True).strip()
    config['input_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(config['annotations']), Path(config['upstream_agent'])]}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    base = upstream_agent(config['upstream_agent'])
    started = time.time()
    started_monotonic = time.monotonic()
    records = []
    with ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
        futures = [pool.submit(run_task, row, config, args.output, base) for row in rows]
        for future in as_completed(futures):
            record = future.result()
            write_json(args.output / f"task-{record['instance_id']}.json", record)
            records.append(record)
            states = {s: sum(r['state'] == s for r in records) for s in {r['state'] for r in records}}
            write_json(args.output / 'summary.json', {
                'started_at_unix': started, 'updated_at_unix': time.time(),
                'elapsed_seconds': time.monotonic() - started_monotonic, 'expected_tasks': len(rows),
                'completed_tasks': len(records), 'states': states,
                'model_calls': sum(len(r['model_calls']) for r in records),
                'prompt_tokens': sum(c.get('usage', {}).get('prompt_tokens', 0) for r in records for c in r['model_calls']),
                'decode_tokens': sum(c.get('usage', {}).get('completion_tokens', 0) for r in records for c in r['model_calls']),
                'all_tasks_terminal': len(records) == len(rows)})
            print(json.dumps({k: record[k] for k in ['instance_id', 'state', 'wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
