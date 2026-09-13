"""Diagnostic only: unchanged SWE renderer, history rewrite, Direct/Host + reference.

Uses recorded tool observations, not an accuracy evaluation or a new harness.
Greedy sampling permits exact comparison; the full500 retains baseline .6.
"""
import argparse
import copy
import json
import os
import time
import uuid
from pathlib import Path

import requests
from transformers import AutoTokenizer
from data.swe_bench_openenv.harness import _render_prompt
from agentic_kv_request import (
    add_agentic_kv_metadata, build_agentic_extra_key,
    confirm_agentic_generation_tool, confirm_agentic_generation_final,
)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--url',default='http://127.0.0.1:23750')
    ap.add_argument('--resume-references', action='store_true')
    ap.add_argument('--baseline',type=Path,default=Path('/tmp/pd-persist/baseline-qwen35-9b-tp1-swe-verified500-colocated-c256-20260910-r2'))
    args=ap.parse_args()
    os.environ['SGLANG_AGENTIC_KV_LIFECYCLE']='true'
    ready=str((args.run_dir/'ready').resolve())
    os.environ['PD_P_READY_DIR']=ready
    tokenizer=AutoTokenizer.from_pretrained('/homes/siqic/Qwen3.5-9B')
    row=json.loads(next((args.baseline/'requests.completed.jsonl').open()))
    trajectory=row['metadata']['openenv_trajectory']
    initial=copy.deepcopy(trajectory['messages'][:2])
    observations=[m for m in trajectory['messages'][2:] if m['role']=='user']
    assert len(observations)>=2
    records=[]

    def send(messages,metadata,generation,reference=False):
        ids=_render_prompt(tokenizer,messages,enable_thinking=True)
        params,rid=add_agentic_kv_metadata(
            dict(temperature=0,top_p=1,top_k=-1,max_new_tokens=8192),
            trajectory_metadata=metadata,generation=generation,tokenizer=tokenizer,
            tool_type='shell',tool_suffix_markers=('```',),terminal_markers=('TASK_COMPLETE',))
        params['custom_params']['agentic_prompt_token_count']=len(ids)
        payload=dict(model='/homes/siqic/Qwen3.5-9B',messages=messages,input_ids=ids,
            temperature=0,top_p=1,top_k=-1,min_p=0,max_completion_tokens=8192,
            chat_template_kwargs=dict(enable_thinking=True),separate_reasoning=True,
            skip_special_tokens=True,no_stop_trim=False,stream=False,
            custom_params=params['custom_params'],extra_key=build_agentic_extra_key(rid,params))
        start=time.time()
        resp=requests.post(args.url+'/v1/chat/completions',json=payload,timeout=600)
        resp.raise_for_status();body=resp.json()
        assert body['choices'][0]['finish_reason']=='stop',body
        return dict(request_id=rid,generation=generation,reference=reference,
                    seconds=time.time()-start,prompt_ids=ids,response=body,
                    request_messages=copy.deepcopy(messages))

    target=args.run_dir/'swe-stable-checkpoint-smoke.json'
    if args.resume_references:
        records=json.loads(target.read_text())
    for delay in (() if args.resume_references else (0.0,3.0)):
        messages=copy.deepcopy(initial)
        metadata={'agentic_request_id':'swe-check-'+uuid.uuid4().hex[:12]}
        for generation in range(3):
            rec=send(messages,metadata,generation)
            rec['tool_delay']=delay
            records.append(rec)
            target.write_text(json.dumps(records,ensure_ascii=False,indent=2))
            message=rec['response']['choices'][0]['message']
            print(json.dumps(dict(generation=generation,delay=delay,seconds=rec['seconds'],
                                  prompt=len(rec['prompt_ids']),usage=rec['response'].get('usage'))),flush=True)
            if generation==2:
                confirm_agentic_generation_final(metadata,generation,p_ready_dir=ready)
            else:
                assert '```' in (message.get('content') or ''),message
                assert confirm_agentic_generation_tool(metadata,generation,p_ready_dir=ready)
                messages.append(dict(role='assistant',content=message.get('content') or '',
                                     reasoning_content=message.get('reasoning_content') or ''))
                messages.append(copy.deepcopy(observations[generation]))
                time.sleep(delay)
    # References run after the timed continuations so they do not consume the
    # one-second Direct admission deadline of any waiting parent generation.
    for rec in records:
        if args.resume_references and 'reference_response' in rec:
            continue
        metadata={'agentic_request_id':'swe-ref-'+uuid.uuid4().hex[:12]}
        ref=send(rec['request_messages'],metadata,0,True)
        confirm_agentic_generation_final(metadata,0,p_ready_dir=ready)
        rec['reference_response']=ref['response']
        rec['reference_request_id']=ref['request_id']
        rec['exact_message_match']=rec['response']['choices'][0]['message']==ref['response']['choices'][0]['message']
        rec['completion_length_match']=rec['response']['usage']['completion_tokens']==ref['response']['usage']['completion_tokens']
        target.write_text(json.dumps(records,ensure_ascii=False,indent=2))
    mismatches=[(r['request_id'],r['generation']) for r in records
                if not (r['exact_message_match'] and r['completion_length_match'])]
    assert not mismatches, mismatches
    print(f'PASS {len(records)} greedy full-recompute references; verify raw state digest/path logs separately',flush=True)


if __name__=='__main__':
    main()
