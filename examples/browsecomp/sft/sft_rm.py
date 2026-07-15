"""High-precision reward used only for offline BrowseComp SFT mining."""

import os

from browsecomp_env import em_score, extract_q_dict
from browsecomp_rm import judge
from slime.utils.types import Sample


async def _score_answer(question: str, label: str, prediction: str) -> tuple[int, str]:
    if em_score(label, prediction):
        return 1, "exact_match"
    first = await judge(question, label, prediction)
    if first != 1:
        return 0, "llm_judge_consensus"
    if os.getenv("BROWSECOMP_JUDGE_CONSENSUS", "1") == "1":
        second = await judge(question, label, prediction)
        return min(first, second), "llm_judge_consensus"
    return first, "llm_judge"


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Return a scalar reward while recording an auditable grading source."""
    if not isinstance(sample, Sample):
        raise TypeError("sample must be an instance of Sample")
    metadata = sample.metadata if sample.metadata is not None else {}
    sample.metadata = metadata
    prediction = (metadata.get("predicted_answer") or "").strip()
    if not prediction:
        metadata.update(grading_source="no_answer", grading_score=0)
        return 0
    question = metadata.get("question")
    label = metadata.get("answer") or sample.label
    assert question and label, "sample.metadata must contain question and answer"

    if "<q1>" in label:
        labels = extract_q_dict(label)
        predictions = extract_q_dict(prediction)
        scores, sources = [], []
        for key, answer in labels.items():
            if key not in predictions:
                scores.append(0)
                sources.append("missing_answer")
                continue
            score, source = await _score_answer(question, answer, predictions[key])
            scores.append(score)
            sources.append(source)
        reward = sum(scores) / len(scores)
        source = "multi:" + ",".join(sources)
    else:
        reward, source = await _score_answer(question, label, prediction)
    metadata.update(grading_source=source, grading_score=reward)
    return reward
