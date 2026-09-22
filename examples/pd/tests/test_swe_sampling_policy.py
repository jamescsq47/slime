from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from inference import make_runtime_args


def workload(harness):
    return SimpleNamespace(datasets=[SimpleNamespace(id="test", harness=harness, options={})])


@pytest.mark.parametrize("field,value", [("temperature", 0), ("top_p", 1), ("top_k", -1)])
def test_swe_wrong_sampling_rejected(field, value):
    cli = Mock(temperature=0.6, top_p=0.95, top_k=20)
    setattr(cli, field, value)
    with pytest.raises(ValueError, match="SWE-bench requires"):
        make_runtime_args(cli, workload("swe_bench_openenv"))


def test_swe_correct_sampling_forwarded_and_other_datasets_unchanged():
    cli = Mock(temperature=0.6, top_p=0.95, top_k=20)
    args = make_runtime_args(cli, workload("swe_bench_openenv"))
    assert (args.rollout_temperature, args.rollout_top_p, args.rollout_top_k) == (0.6, 0.95, 20)
    cli.temperature, cli.top_p, cli.top_k = 0, 1, -1
    args = make_runtime_args(cli, workload("browsecomp"))
    assert (args.rollout_temperature, args.rollout_top_p, args.rollout_top_k) == (0, 1, -1)
