import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).parents[2] / "slime/backends/megatron_utils/cp_utils.py"
_SPEC = importlib.util.spec_from_file_location("cp_utils_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cp_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cp_utils)


def test_sample_mean_can_exclude_fully_masked_samples(monkeypatch):
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 1)

    masks = [torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])]
    values = torch.tensor([1.2, 0.8, 9.0, 9.0])

    legacy_reducer = cp_utils.get_sum_of_sample_mean([2, 2], [2, 2], masks)
    metric_reducer = cp_utils.get_sum_of_sample_mean([2, 2], [2, 2], masks, skip_empty_samples=True)

    # The legacy reducer turns the fully masked sample into a zero observation
    # when the common batch denominator is applied by the logging path.
    assert torch.allclose(legacy_reducer(values) / len(masks), torch.tensor(0.5))

    # A mismatch metric uses its own active-sample denominator, so the valid
    # sample's mean is reported unchanged.
    active_samples = sum(mask.sum() > 0 for mask in masks)
    assert torch.allclose(metric_reducer(values) / active_samples, torch.tensor(1.0))
    assert torch.allclose(active_samples / len(masks), torch.tensor(0.5))


def test_task_metric_uses_only_its_own_effective_samples(monkeypatch):
    monkeypatch.setattr(cp_utils.mpu, "get_context_parallel_world_size", lambda: 1)

    task_types = ["math", "qa", "qa"]
    masks = [torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0])]
    tis = torch.tensor([1.2, 9.0, 0.8])

    def task_mean(task_type: str) -> torch.Tensor:
        task_masks = [mask if sample_task == task_type else torch.zeros_like(mask) for sample_task, mask in zip(task_types, masks)]
        reducer = cp_utils.get_sum_of_sample_mean([1, 1, 1], [1, 1, 1], task_masks, skip_empty_samples=True)
        active_samples = sum(mask.sum() > 0 for mask in task_masks)
        return reducer(tis) / active_samples

    assert torch.allclose(task_mean("math"), torch.tensor(1.2))
    assert torch.allclose(task_mean("qa"), torch.tensor(0.8))
