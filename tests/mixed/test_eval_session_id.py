import asyncio
from argparse import Namespace
from types import SimpleNamespace

from slime.rollout import sglang_rollout
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.types import Sample


def test_eval_rollout_assigns_unique_session_id_per_trajectory(monkeypatch):
    dataset_cfg = EvalDatasetConfig(
        name="terminal",
        path="unused.jsonl",
        input_key="prompt",
        label_key=None,
        metadata_key="metadata",
        n_samples_per_eval_prompt=2,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_response_len=32,
        eval_reward_key="score",
    )
    args = Namespace(
        group_rm=False,
        hf_checkpoint="unused-model",
        apply_chat_template=False,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        sglang_enable_deterministic_inference=False,
        eval_reward_key="score",
        reward_key=None,
    )
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    prompt = Sample(prompt="test", session_id="copied-session")
    monkeypatch.setitem(
        sglang_rollout.EVAL_PROMPT_DATASET,
        cache_key,
        SimpleNamespace(samples=[prompt]),
    )

    seen = []

    async def fake_generate_and_rm(args, sample, sampling_params, evaluation):
        assert evaluation is True
        seen.append(sample)
        sample.reward = {"score": 0.0}
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(sglang_rollout, "generate_and_rm", fake_generate_and_rm)

    result = asyncio.run(sglang_rollout.eval_rollout_single_dataset(args, 0, dataset_cfg))

    session_ids = [sample.session_id for sample in seen]
    assert len(session_ids) == 2
    assert all(session_ids)
    assert len(set(session_ids)) == 2
    assert "copied-session" not in session_ids
    assert result["terminal"]["rewards"] == [0.0, 0.0]
