"""Memory-efficient loss for BrowseComp trajectory SFT.

The generic RL log-prob path evaluates vocabulary cross entropy for every
token after the first assistant turn, including masked search-page payloads.
BrowseComp supervises only assistant tokens, so select those positions before
the vocabulary-parallel CE.  This removes a multi-GiB temporary on TP=2.
"""

from __future__ import annotations

import torch
from megatron.core import mpu

from slime.backends.megatron_utils.loss import get_responses
from slime.utils.ppo_utils import calculate_log_probs_and_entropy


def sft_masked_loss(args, batch, logits, sum_of_sample_mean):
    response_logits_and_tokens = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens"),
    )

    full_log_probs = []
    selected_tokens = 0
    for (response_logits, response_tokens), loss_mask in zip(
        response_logits_and_tokens, batch["loss_masks"], strict=True
    ):
        selected = loss_mask.bool()
        if selected.numel() != response_logits.size(0):
            raise ValueError(
                f"SFT response mask/logit mismatch: {selected.numel()} != {response_logits.size(0)}"
            )
        selected_tokens += int(selected.sum())
        selected_log_probs, _ = calculate_log_probs_and_entropy(
            response_logits[selected],
            response_tokens[selected],
            mpu.get_tensor_model_parallel_group(),
            with_entropy=False,
            chunk_size=args.log_probs_chunk_size,
        )
        # Keep the generic reducer's per-sample/per-token normalization exact.
        full = response_logits.new_zeros(response_logits.size(0))
        full = full.masked_scatter(selected, selected_log_probs.reshape(-1))
        full_log_probs.append(full)

    if selected_tokens == 0:
        loss = 0 * logits.sum()
    else:
        loss = -sum_of_sample_mean(torch.cat(full_log_probs))
    return loss, {"loss": loss.detach()}
