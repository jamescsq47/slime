import json
from pathlib import Path

import pytest

from data.bfcl_multi_turn import probe

ROOT = Path('/homes/siqic/data/BFCL-upstream/berkeley-function-call-leaderboard')


def sample():
    row = json.loads((ROOT/'bfcl_eval/data/BFCL_v4_multi_turn_base.json').read_text().splitlines()[0])
    gold = json.loads((ROOT/'bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json').read_text().splitlines()[0])['ground_truth']
    return row, gold


@pytest.mark.parametrize('text', ["[x(a=__import__('os'))]", '[x(**{})]', '[os.x()]', '[unknown()]', '[1]'])
def test_reject_code(text):
    with pytest.raises((ValueError, SyntaxError)):
        probe.decode_calls(text, {'x'})


def test_parse():
    assert probe.decode_calls('<think>secret</think>[x(a=[1, 2])]', {'x'}) == ['x(a=[1, 2])']
    assert probe.decode_calls('Done.', {'x'}) == []
    assert probe.common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert probe.decode_calls('[x(a=true)]', {'x'}) == ["x(a='true')"]


def test_real_tokenizer_wire_format():
    tokenizer = probe.AutoTokenizer.from_pretrained('/dataset/model/qwen3.5/Qwen3.5-9B')
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Hi'}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    json.dumps({'input_ids': ids})


def test_official_gold_replay(tmp_path):
    row, gold = sample()
    ex = probe.ContainerPython(Path(probe.__file__).parent, 30, image='bfcl-multiturn-probe:local')
    try:
        probe.init_executor(ex, row)
        for turn in gold:
            for call in turn:
                assert probe.execute_call(ex, call)['seconds'] >= 0
        assert probe.score(ex, row, [[turn] for turn in gold], gold)['valid']
        assert not probe.score(ex, row, [[] for _ in gold], gold)['valid']
    finally:
        ex.cleanup()
    assert not ex.cleanup_errors


def test_loop_and_prefix_audit(tmp_path):
    row, gold = sample()
    cfg = dict(upstream=str(ROOT), image='bfcl-multiturn-probe:local', max_steps_per_turn=20,
               max_new_tokens=8192, context_length=40960)
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1] * len(messages)
    def model(ids, messages):
        assert 'ground_truth' not in str(messages)
        return dict(text='Done.', output_ids=[2])
    result = probe.run_task(row, gold, cfg, tmp_path, Tokenizer(), model)
    assert result['state'] == 'completed', result
    assert len(result['calls']) == len(row['question'])
    assert len(result['prefix_checks']) == len(row['question'])-1
    assert len(result['calls'][-1]['messages']) > len(result['calls'][0]['messages'])
    assert not result['score']['valid']
    assert not result['cleanup_errors']
    def malformed(ids, messages):
        return dict(text='[ls(a=)]', output_ids=[2])
    bad = probe.run_task(row, gold, cfg, tmp_path, Tokenizer(), malformed)
    assert bad['state'] == 'completed'
    assert len(bad['parse_errors']) == len(row['question'])
    assert not bad['cleanup_errors']
    cfg['context_length'] = 1
    failed = probe.run_task(row, gold, cfg, tmp_path, Tokenizer(), model)
    assert failed['state'] == 'error' and 'Context budget' in failed['error']
    assert not failed['cleanup_errors']
