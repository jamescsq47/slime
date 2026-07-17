"""SFT-only rollout with bounded BrowseComp evidence contexts.

The raw successful trajectories can exceed 30k tokens.  FlashAttention 2 on
Ampere has repeatedly stalled on that tail, so compact old search/open-page
payloads while retaining every trajectory, tool call, assistant turn, and the
final answer.  This module is intentionally local to the BrowseComp SFT job.
"""

from __future__ import annotations

import copy
import logging
import os

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_processor, load_tokenizer

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None
SAMPLE_PRINTED = False


def _is_evidence_message(message: dict) -> bool:
    content = message.get("content", "")
    return message.get("role") == "user" and isinstance(content, str) and content.startswith(
        ("[Search Results for ", "[Opened Page Content]")
    )


def _compact_messages(messages: list[dict], tools, mask_generator, max_tokens: int):
    """Replace oldest evidence payloads until the rendered chat fits."""
    compacted = copy.deepcopy(messages)
    token_ids, loss_mask = mask_generator.get_loss_mask(compacted, tools=tools)
    original_length = len(token_ids)
    if original_length <= max_tokens:
        return compacted, token_ids, loss_mask, 0

    candidates = [i for i, message in enumerate(compacted[:-1]) if _is_evidence_message(message)]
    for index in candidates:
        content = compacted[index]["content"]
        first_line = content.splitlines()[0]
        compacted[index]["content"] = f"{first_line}\n[Earlier evidence payload omitted for SFT context budget.]"
        token_ids, loss_mask = mask_generator.get_loss_mask(compacted, tools=tools)
        if len(token_ids) <= max_tokens:
            break

    if len(token_ids) > max_tokens:
        raise ValueError(
            f"SFT trajectory remains too long after evidence compaction: {len(token_ids)} > {max_tokens}"
        )
    return compacted, token_ids, loss_mask, original_length - len(token_ids)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR, MASK_GENERATOR, SAMPLE_PRINTED
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)
    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)

    max_tokens = int(os.environ.get("SFT_MAX_SEQ_LEN", "16384"))
    samples = data_buffer.get_samples(args.rollout_batch_size)
    for wrapped_sample in samples:
        (sample,) = wrapped_sample
        metadata = sample.metadata or {}
        exact_tokens = metadata.get("pretokenized_tokens")
        exact_mask = metadata.get("pretokenized_loss_mask")
        exact_mode = exact_tokens is not None or exact_mask is not None
        if exact_mode:
            # Math rollouts are exported with their original RL mask.  In
            # particular, interpreter observations remain unsupervised.
            if not isinstance(exact_tokens, list) or not isinstance(exact_mask, list):
                raise ValueError("incomplete pretokenized SFT fields")
            if not exact_mask or len(exact_tokens) < len(exact_mask) or len(exact_tokens) > max_tokens:
                raise ValueError("invalid or over-budget pretokenized SFT trajectory")
            token_ids, loss_mask, removed = exact_tokens, exact_mask, 0
            response_length = len(loss_mask)
        else:
            messages = sample.prompt
            tools = metadata.get("tools")
            compacted, token_ids, loss_mask, removed = _compact_messages(
                messages, tools, MASK_GENERATOR, max_tokens
            )
            response_length = MASK_GENERATOR.get_response_lengths([loss_mask])[0]
        if not exact_mode and len(token_ids) != len(loss_mask):
            # `loss_mask` is response-relative for pretokenized rollouts.
            raise ValueError(f"mismatched SFT tokens/mask: {len(token_ids)} != {len(loss_mask)}")
        if response_length == 0:
            raise ValueError("SFT trajectory has no supervised assistant tokens")

        sample.tokens = token_ids
        sample.response_length = response_length
        sample.reward = 0
        sample.loss_mask = loss_mask[-response_length:]
        sample.metadata["sft_original_token_length"] = len(token_ids) + removed
        sample.metadata["sft_compacted_tokens"] = removed

        if not SAMPLE_PRINTED:
            logger.info(
                "BrowseComp SFT sample: original_tokens=%d train_tokens=%d compacted_tokens=%d max_tokens=%d",
                len(token_ids) + removed,
                len(token_ids),
                removed,
                max_tokens,
            )
            SAMPLE_PRINTED = True

    return samples
