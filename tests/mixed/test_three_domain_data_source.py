import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from slime.utils.types import Sample

MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "mixed" / "custom_data_source.py"
spec = importlib.util.spec_from_file_location("mixed_data_source_under_test", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _source(math_ratio, terminal_ratio):
    source = object.__new__(module.CustomDataSource)
    source.math_ratio = math_ratio
    source.terminal_ratio = terminal_ratio
    source.args = SimpleNamespace(rollout_shuffle=False, rollout_seed=7)
    return source


def _sample(task_type):
    return Sample(metadata={"task_type": task_type})


def _alternating_source(
    math_batches=2,
    qa_batches=2,
    terminal_batches=2,
    order=3,
    start_task=None,
):
    source = object.__new__(module.CustomDataSource)
    source.args = SimpleNamespace(
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        rollout_shuffle=False,
        current_policy_version=0,
    )
    source.math_batches_per_cycle = math_batches
    source.qa_batches_per_cycle = qa_batches
    source.terminal_batches_per_cycle = terminal_batches
    source.batch_alternation_order = order
    source.batch_alternation_start_task = start_task
    source.math_samples = [_sample("math") for _ in range(8)]
    source.qa_samples = [_sample("qa") for _ in range(8)]
    source.terminal_samples = [_sample("terminal") for _ in range(8)]
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    source._init_batch_alternator()
    return source


def test_three_domain_mixing_recycles_to_requested_ratio():
    mixed = _source(0.4, 0.2)._mix_samples(
        [_sample("math")],
        [_sample("qa")],
        [_sample("terminal")],
    )
    counts = {
        task: sum(sample.metadata["task_type"] == task for sample in mixed) for task in ("math", "qa", "terminal")
    }
    assert counts == {"math": 2, "qa": 2, "terminal": 1}


def test_terminal_ratio_zero_preserves_two_domain_mix():
    mixed = _source(0.5, 0.0)._mix_samples(
        [_sample("math")],
        [_sample("qa")],
        [],
    )
    assert sorted(sample.metadata["task_type"] for sample in mixed) == ["math", "qa"]


def test_terminal_jsonl_loads_without_supervised_label(tmp_path):
    path = tmp_path / "terminal.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": [{"role": "system", "content": "terminal agent"}],
                "metadata": {"task_id": "task-a"},
            }
        )
        + "\n"
    )
    args = SimpleNamespace(
        rollout_max_prompt_len=4096,
        input_key="prompt",
        multimodal_keys=None,
        metadata_key="metadata",
        tool_key=None,
        rollout_seed=1,
    )
    samples = _source(0.0, 1.0)._load_terminal_data(
        args,
        tokenizer=None,
        processor=None,
        terminal_path=str(path),
    )
    assert len(samples) == 1
    assert samples[0].label is None
    assert samples[0].metadata["task_id"] == "task-a"
    assert samples[0].metadata["task_type"] == "terminal"


def test_three_domain_batch_alternation_uses_complete_fixed_cycles():
    source = _alternating_source()
    assert source.batch_sequence == [
        "qa",
        "qa",
        "math",
        "math",
        "terminal",
        "terminal",
    ]

    dispatched = []
    for _ in range(8):
        groups = source._get_samples_alternating(2)
        assert len(groups) == 2
        assert all(len(group) == 1 for group in groups)
        dispatched.append(groups[0][0].metadata["task_type"])
    assert dispatched == [
        "qa",
        "qa",
        "math",
        "math",
        "terminal",
        "terminal",
        "qa",
        "qa",
    ]
    # QA consumed four prompts in the first cycle and continues from there;
    # the scheduling-cycle boundary does not reset it to the dataset head.
    assert source.qa_offset == 0


def test_terminal_quota_zero_preserves_two_domain_start_order():
    source = _alternating_source(terminal_batches=0)
    assert source.batch_sequence[:4] == ["qa", "qa", "math", "math"]


def test_all_six_three_domain_orders():
    expected_orders = {
        1: ("math", "qa", "terminal"),
        2: ("math", "terminal", "qa"),
        3: ("qa", "math", "terminal"),
        4: ("qa", "terminal", "math"),
        5: ("terminal", "math", "qa"),
        6: ("terminal", "qa", "math"),
    }
    for order, expected in expected_orders.items():
        source = _alternating_source(
            math_batches=1,
            qa_batches=1,
            terminal_batches=1,
            order=order,
        )
        assert source.batch_sequence == list(expected)


def test_legacy_start_task_remains_supported_without_explicit_order():
    source = _alternating_source(order=None, start_task="qa")
    assert source.batch_sequence == [
        "qa",
        "qa",
        "math",
        "math",
        "terminal",
        "terminal",
    ]


def test_token_load_order_uses_warmup_thresholds_and_sliding_window():
    assert module._choose_token_load_order(1_000, [100] * 7) == ("random", None, None)

    order, reference, ratio = module._choose_token_load_order(111, [100] * 8)
    assert order == "lpt"
    assert reference == 100
    assert ratio == 1.11

    order, reference, ratio = module._choose_token_load_order(89, [100] * 8)
    assert order == "spt"
    assert reference == 100
    assert ratio == 0.89

    # Only the most recent 16 entries form the baseline.
    order, reference, ratio = module._choose_token_load_order(100, [10_000] + [100] * 16)
    assert order == "random"
    assert reference == 100
    assert ratio == 1.0


def test_token_load_reordering_preserves_window_and_prioritizes_domains():
    prompts = [
        Sample(prompt=f"{task}-{index}", metadata={"task_type": task})
        for index, task in enumerate(["qa", "terminal", "math", "qa", "terminal", "math"])
    ]

    lpt = module._reorder_prompt_window(prompts, "lpt", seed=7)
    spt = module._reorder_prompt_window(prompts, "spt", seed=7)

    assert sorted(sample.prompt for sample in lpt) == sorted(sample.prompt for sample in prompts)
    assert [sample.metadata["task_type"] for sample in lpt] == [
        "terminal",
        "terminal",
        "math",
        "math",
        "qa",
        "qa",
    ]
    assert [sample.metadata["task_type"] for sample in spt] == [
        "qa",
        "qa",
        "math",
        "math",
        "terminal",
        "terminal",
    ]


def test_adaptive_update_counts_full_tokens_and_prepares_exactly_next_batch():
    source = object.__new__(module.CustomDataSource)
    source.args = SimpleNamespace(
        rollout_batch_size=32,
        rollout_seed=13,
        rollout_shuffle=False,
    )
    source.token_load_adaptive_reordering = True
    source.train_token_history = [100] * 8
    source.pending_prompt_samples = []
    source.token_load_order = "random"
    source.token_load_update_count = 8
    source._sample_lock = threading.RLock()
    source.epoch_id = 0
    source.sample_offset = 0
    source.origin_samples = [
        Sample(prompt=f"{task}-{index}", metadata={"task_type": task})
        for index, task in enumerate((["qa", "math", "terminal"] * 11)[:32])
    ]
    source.samples = list(source.origin_samples)

    # Loss-masked prompt/tool tokens still contribute to train compute.
    train_samples = [Sample(tokens=list(range(120)), loss_mask=[0] * 120)]
    metrics = source.update_token_load_adaptive_order(train_samples)

    assert metrics["perf/train_batch_full_tokens"] == 120
    assert metrics["perf/train_token_ratio"] == 1.2
    assert metrics["perf/token_load_order"] == module.TOKEN_LOAD_ORDER_CODES["lpt"]
    assert len(source.pending_prompt_samples) == 32
    assert source.sample_offset == 32
    task_order = [sample.metadata["task_type"] for sample in source.pending_prompt_samples]
    assert task_order == sorted(task_order, key={"terminal": 0, "math": 1, "qa": 2}.get)
    assert sorted(sample.prompt for sample in source.pending_prompt_samples) == sorted(
        sample.prompt for sample in source.origin_samples
    )


def test_adaptive_reordering_checkpoint_restores_history_and_pending_indices(tmp_path):
    origin_samples = [
        Sample(prompt=f"prompt-{index}", metadata={"task_type": "math"})
        for index in range(4)
    ]
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_shuffle=False,
        save=str(tmp_path),
        load=str(tmp_path),
    )

    source = object.__new__(module.CustomDataSource)
    source.args = args
    source._sample_lock = threading.RLock()
    source.batch_alternation = False
    source.token_load_adaptive_reordering = True
    source.sample_offset = 3
    source.epoch_id = 2
    source.sample_group_index = 12
    source.sample_index = 96
    source.metadata = {}
    source.origin_samples = origin_samples
    source.train_token_history = [100, 110]
    source.pending_prompt_samples = [origin_samples[2], origin_samples[0]]
    source.token_load_order = "lpt"
    source.token_load_update_count = 2
    source.save(7)

    restored = object.__new__(module.CustomDataSource)
    restored.args = args
    restored._sample_lock = threading.RLock()
    restored.batch_alternation = False
    restored.token_load_adaptive_reordering = True
    restored.origin_samples = origin_samples
    restored.samples = list(origin_samples)
    restored.train_token_history = []
    restored.pending_prompt_samples = []
    restored.token_load_order = "random"
    restored.token_load_update_count = 0
    restored.load(7)

    assert restored.train_token_history == [100, 110]
    assert [sample.prompt for sample in restored.pending_prompt_samples] == [
        "prompt-2",
        "prompt-0",
    ]
    assert restored.token_load_order == "lpt"
    assert restored.token_load_update_count == 2
