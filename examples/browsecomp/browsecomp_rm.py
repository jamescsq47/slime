# BrowseComp-Plus LLM-judge reward model.
#
# Grader prompt + parsing/judge logic ported from the Context-Folding /
# FoldAgent open-source re-implementation (https://arxiv.org/abs/2510.11967,
# envs/local_search.py), Apache License 2.0 (same as this repo).
# See README "Attribution & License".

"""LLM-judge reward model for BrowseComp-Plus RL training.

Plugged into slime via:
  --custom-rm-path browsecomp_rm.reward_func

Grading pipeline (ported from FoldAgent envs/local_search.py):
  1. lenient exact-match fast path (free),
  2. official BrowseComp grader prompt scored by an OpenAI-compatible
     LLM judge (up to 3 attempts to get a parseable verdict),
  3. if the judge said no but the prediction is a relaxed-EM superset of
     the ground truth, a stronger fallback judge gets a second opinion.

Environment variables:
  GRADER_API_KEY / OPENAI_API_KEY      judge API key (required)
  GRADER_BASE_URL / OPENAI_BASE_URL    judge endpoint (default: OpenAI)
  GRADER_API_VERSION                   set => Azure-style endpoint
  GRADER_MODEL                         primary judge model
  GRADER_FALLBACK_MODEL                second-opinion judge model
  GRADER_REQUEST_TIMEOUT               per-request timeout seconds (default 300)
"""

import asyncio
import difflib
import logging
import os
import random
import re
import unicodedata
from collections import Counter

from browsecomp_env import em_score, extract_q_dict

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, contains all the essential information from [correct_answer], is equivalent despite minor wording/order differences (such as name order, inclusion or omission of middle names/initials, common honorifics, standard shortenings of first names, inclusion/omission of non-contradictory date parts like year, minor articles like "a"/"the", extra descriptive context, non-essential descriptive prefixes/suffixes such as "Restaurant", "Inc.", "Ltd.", or sports suffixes like "FC", "CF", "SC", inclusion/omission of subtitles in titles, minor spacing/punctuation differences — including presence/absence of quotation marks, interchangeable punctuation such as ":" / "-" / "–", case-only differences, or presence/absence of diacritics), or is within a small margin of error for numerical problems. Answer 'no' only if the extracted answer is factually incorrect, missing essential identifying information, or contradicts the [correct_answer].

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


class GraderFatalError(RuntimeError):
    """Unrecoverable grader misconfiguration (bad key/model/endpoint).
    Raised instead of silently scoring 0 so training does not proceed on
    garbage rewards."""


def parse_judge_response(judge_response: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    patterns = [
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
        r"extracted_final_answer:\s*(.*?)(?=\n|$)",
    ]
    for p in patterns:
        if m := re.search(p, judge_response, re.IGNORECASE | re.DOTALL):
            result["extracted_final_answer"] = m.group(1).strip()
            break

    for p in [
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
    ]:
        if m := re.search(p, judge_response, re.IGNORECASE | re.DOTALL):
            result["reasoning"] = m.group(1).strip()
            break

    for p in [r"\*\*correct:\*\*\s*(yes|no)", r"\*\*correct\*\*:\s*(yes|no)", r"correct:\s*(yes|no)"]:
        if m := re.search(p, judge_response, re.IGNORECASE):
            result["correct"] = m.group(1).lower() == "yes"
            break

    for p in [
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?",
        r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?",
        r"confidence:\s*(\d+(?:\.\d+)?)\s*%?",
    ]:
        if m := re.search(p, judge_response, re.IGNORECASE):
            result["confidence"] = min(float(m.group(1)), 100.0)
            break

    if result["correct"] is None:
        result["parse_error"] = True

    return result


def relaxed_em(label: str, pred: str) -> bool:
    deacc = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    norm = lambda s: re.sub(
        r"\s+",
        " ",
        re.sub(r"\s*\([^)]*\)\s*", " ", re.sub(r"[“”\"'`]+", "", re.sub(r"[:–—\-_/.,;!()?]+", " ", deacc(s).lower()))),
    ).strip()
    strip = lambda s: re.sub(r"\s+", "", norm(s))
    if not label or not pred:
        return False
    A, B = strip(label), strip(pred)
    if A == B or A in B or B in A:
        return True
    if difflib.SequenceMatcher(None, A, B).ratio() >= 0.9:
        return True
    ca, cb = Counter(A), Counter(B)
    if sum((ca & cb).values()) / min(len(A), len(B) or 1) >= 0.9:
        return True
    return False


async def call_grader(messages, model=None, max_retries=8):
    from openai import (
        APITimeoutError,
        AsyncAzureOpenAI,
        AsyncOpenAI,
        AuthenticationError,
        BadRequestError,
        NotFoundError,
        PermissionDeniedError,
    )

    if model is None:
        model = os.getenv("GRADER_MODEL")
        if not model:
            # The Gemini OpenAI-compatible endpoint 404s on OpenAI-only names,
            # so pick a backend-compatible default.
            base_url = os.getenv("GRADER_BASE_URL") or os.getenv("OPENAI_BASE_URL", "") or ""
            model = "gemini-2.0-flash" if "generativelanguage.googleapis.com" in base_url else "gpt-4o-mini"
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    grader_api_key = os.getenv("GRADER_API_KEY") or os.getenv("OPENAI_API_KEY")
    grader_base_url = os.getenv("GRADER_BASE_URL") or os.getenv("OPENAI_BASE_URL", None)
    grader_api_version = os.getenv("GRADER_API_VERSION")  # set => Azure-style endpoint
    if not grader_api_key:
        raise GraderFatalError(
            "Grader requires GRADER_API_KEY (or OPENAI_API_KEY) but neither is set; refusing to silently score 0."
        )
    per_req_timeout = float(os.getenv("GRADER_REQUEST_TIMEOUT", "300"))
    if grader_api_version:
        client = AsyncAzureOpenAI(
            api_key=grader_api_key,
            azure_endpoint=grader_base_url,
            api_version=grader_api_version,
            timeout=per_req_timeout,
        )
    else:
        client = AsyncOpenAI(api_key=grader_api_key, base_url=grader_base_url, timeout=per_req_timeout)

    try:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(model=model, messages=messages)
                # Some OpenAI-compatible endpoints (e.g. Gemini) can return no
                # choices or a None message (safety-blocked / reasoning-only);
                # treat that as an empty judgement rather than crashing.
                choice = resp.choices[0] if resp.choices else None
                message = getattr(choice, "message", None) if choice is not None else None
                return (getattr(message, "content", None) or "") if message is not None else ""
            except (AuthenticationError, PermissionDeniedError, NotFoundError) as e:
                raise GraderFatalError(
                    f"Grader API rejected the request with a non-retryable error ({type(e).__name__}): {e}. "
                    f"Check GRADER_API_KEY, GRADER_BASE_URL, and GRADER_MODEL."
                ) from e
            except BadRequestError as e:
                msg = str(e).lower()
                if (
                    "model" in msg and ("not found" in msg or "unexpected" in msg or "invalid" in msg)
                ) or "authorization" in msg:
                    raise GraderFatalError(
                        f"Grader API rejected the request with a non-retryable bad-request error: {e}. "
                        f"Check GRADER_MODEL and GRADER_API_KEY."
                    ) from e
                if attempt == max_retries - 1:
                    raise GraderFatalError(f"Grader API bad request after {max_retries} attempts: {e}") from e
                await asyncio.sleep(min(60.0, 2.0**attempt) + random.uniform(0, 1.5))
            except APITimeoutError as e:
                # Under rate-limit pressure timeouts repeat; back off long
                # enough to wait out the window instead of hammering it.
                if attempt == max_retries - 1:
                    raise GraderFatalError(
                        f"Grader API timed out after {max_retries} attempts "
                        f"(per-request timeout={per_req_timeout}s): {e}"
                    ) from e
                await asyncio.sleep(min(120.0, 5.0 * (2.0**attempt)) + random.uniform(0, 2.0))
            except Exception as e:
                if attempt == max_retries - 1:
                    raise GraderFatalError(f"Grader API failed after {max_retries} attempts: {e}") from e
                await asyncio.sleep(min(60.0, 2.0**attempt) + random.uniform(0, 1.5))
        raise GraderFatalError("Grader API exhausted retries without a response")
    finally:
        await client.close()


async def judge(question: str, correct_answer: str, predicted_answer: str) -> int:
    # Patch known BrowseComp ground-truth typos (kept reversed to avoid
    # polluting plain-text search; ported from FoldAgent).
    correct_answer = "ttellomS saiboT"[::-1] if "tellomS saiboT"[::-1] in correct_answer else correct_answer
    correct_answer = (
        "yayhdapottahC najnarawsiB"[::-1] if "yayhdapattahC najnarawsiB"[::-1] in correct_answer else correct_answer
    )
    predicted_answer = (
        "yrtnuoC a fo htaP ehT :sedirelC sokfalG"[::-1]
        if "yrtnuoC a fo htaP ehT :sedirelC socfalG"[::-1] in predicted_answer
        else predicted_answer
    )

    if em_score(correct_answer, predicted_answer):
        return 1
    if len(predicted_answer.strip()) == 0:
        return 0

    judge_prompt = GRADER_TEMPLATE.format(
        question=question, response=predicted_answer, correct_answer=correct_answer
    )
    messages = [{"role": "user", "content": judge_prompt}]
    score = 0
    for _ in range(3):
        response = await call_grader(messages)
        grade_report = parse_judge_response(response)
        if grade_report["parse_error"] or grade_report.get("correct") is None:
            continue
        score = int(bool(grade_report["correct"]))
        break

    if score == 0 and relaxed_em(correct_answer, predicted_answer):
        # Second-opinion path: the primary judge said no but the prediction is
        # a relaxed superset of the ground truth — consult a stronger judge.
        fallback_model = os.getenv("GRADER_FALLBACK_MODEL")
        if not fallback_model:
            base_url = os.getenv("GRADER_BASE_URL") or os.getenv("OPENAI_BASE_URL", "") or ""
            primary = os.getenv("GRADER_MODEL", "") or ""
            if "generativelanguage.googleapis.com" in base_url or primary.lower().startswith("gemini"):
                fallback_model = "gemini-2.5-pro"
            else:
                fallback_model = "gpt-4.1"
        try:
            response = await call_grader(messages, model=fallback_model)
            grade_report = parse_judge_response(response)
            if not grade_report["parse_error"] and grade_report.get("correct") is not None:
                score = int(bool(grade_report["correct"]))
        except GraderFatalError as e:
            logger.warning(
                "relaxed-EM fallback grader %r failed: %s. Keeping primary score=0. "
                "Set GRADER_FALLBACK_MODEL to a backend-compatible model.",
                fallback_model,
                e,
            )

    return score


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Reward for one BrowseComp rollout. 1 if the submitted answer matches
    the ground truth per the official BrowseComp judge, else 0. Rollouts that
    never submitted a valid finish call get 0 without a judge call."""
    if not isinstance(sample, Sample):
        raise TypeError("sample must be an instance of Sample")

    metadata = sample.metadata or {}
    predicted_answer = metadata.get("predicted_answer")
    if not predicted_answer or not predicted_answer.strip():
        return 0

    question = metadata.get("question")
    label_answer = metadata.get("answer") or sample.label
    assert question and label_answer, "sample.metadata must contain 'question' and 'answer'"

    em_only = os.getenv("BROWSECOMP_EM_ONLY_REWARD", "0") == "1"

    if "<q1>" in label_answer:
        # Multi-question labels: average the per-question judgements.
        label_answer_dict = extract_q_dict(label_answer)
        predicted_answer_dict = extract_q_dict(predicted_answer)
        all_reward = []
        for k in label_answer_dict:
            if k not in predicted_answer_dict:
                all_reward.append(0)
            elif em_only:
                all_reward.append(int(em_score(label_answer_dict[k], predicted_answer_dict[k])))
            else:
                all_reward.append(await judge(question, label_answer_dict[k], predicted_answer_dict[k]))
        return sum(all_reward) / len(all_reward)

    if em_only:
        return int(em_score(label_answer, predicted_answer))
    return await judge(question, label_answer, predicted_answer)
