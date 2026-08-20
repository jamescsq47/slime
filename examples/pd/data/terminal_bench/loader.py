"""Load Terminal-Bench task ids; the environment supplies each instruction."""

from __future__ import annotations

import logging

from slime.utils.data import Dataset
from slime.utils.types import Sample

from data.api import LoadContext
from data.config import DatasetSpec


LOG = logging.getLogger(__name__)


def load_samples(context: LoadContext, dataset: DatasetSpec) -> list[Sample]:
    options = dataset.options
    source = Dataset(
        dataset.path,
        tokenizer=context.tokenizer,
        processor=context.processor,
        max_length=context.args.rollout_max_prompt_len,
        prompt_key=str(options.get("prompt_key", context.args.input_key)),
        multimodal_keys=None,
        label_key=None,
        metadata_key=str(options.get("metadata_key", context.args.metadata_key)),
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        seed=context.args.rollout_seed,
    )
    for sample in source.origin_samples:
        metadata = sample.metadata or {}
        if not metadata.get("task_id"):
            raise ValueError("every Terminal-Bench row requires metadata.task_id")
        sample.metadata = metadata
    LOG.info("Loaded %d Terminal-Bench samples from %s", len(source.origin_samples), dataset.path)
    return source.origin_samples
