# BrowseComp-Plus tool environment (search / open_page / finish).
#
# Ported/adapted from the Context-Folding / FoldAgent open-source
# re-implementation (https://arxiv.org/abs/2510.11967, envs/local_search.py),
# Apache License 2.0 (same as this repo). See README "Attribution & License".
#
# Text-format function-call parser + per-rollout action state machine over the
# local retrieval server (search_server.py), which exposes:
#   POST /search {"query": str, "k": int}     -> {"results": [{docid, url, text}, ...]}
#   POST /open   {"url": str | "docid": str}  -> {"results": [{docid, url, text}, ...]}
# Its URL is read from the LOCAL_SEARCH_URL environment variable.

import ast
import asyncio
import json
import os
import re
import unicodedata
from collections import Counter

import httpx


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def keep_first_n_words(text: str, n: int = 1000) -> str:
    if not text:
        return ""
    count = 0
    for m in re.finditer(r"\S+", text):
        count += 1
        if count == n:
            return text[: m.end()] + "\n[Document is truncated.]"
    return text


def em_score(label: str, pred: str) -> bool:
    """Lenient exact-match used as the LLM-judge fast path and the
    no-search answer guard. Ported verbatim from FoldAgent."""
    ign = {"a", "an", "the", "of", "on", "in", "and", "&", "for", "to", "by", "with"}
    deacc = lambda s: "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

    def norm(s: str) -> str:
        s = deacc(s).lower()
        s = re.sub(r"\s*\([^)]*\)\s*", " ", s)  # drop parenthetical qualifiers: (Egypt), (US), etc.
        s = re.sub(r"[“”\"'`]+", "", s)  # drop quotes
        s = re.sub(r"[:–—\-_/.,;!()?]+", " ", s)  # unify punctuation to spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s

    strip = lambda s: re.sub(r"\s+", "", norm(s))
    toks = lambda s: [t for t in norm(s).split() if t not in ign and not re.fullmatch(r"\d{4}", t)]
    if strip(label) == strip(pred):
        return True
    lt, pt = toks(label), toks(pred)
    if not lt or not pt:
        return False
    if Counter(lt) == Counter(pt):
        return True
    if len(lt) >= 2 and len(pt) >= 2 and lt[-1] == pt[-1]:
        f1, f2 = lt[0], pt[0]
        if f1 == f2 or (min(len(f1), len(f2)) >= 4 and (f1.startswith(f2) or f2.startswith(f1))):
            return True
    head = lambda s: strip(re.split(r"[:–—-]", norm(s), 1)[0])
    if head(label) == head(pred):
        return True
    return False


def extract_q_dict(s: str) -> dict[str, str]:
    return {k: v.strip() for k, v in re.findall(r"<(q\d+)>(.*?)</\1>", s, flags=re.S)}


class SearchRequestError(RuntimeError):
    """The search backend rejected a model-generated request (4xx).
    Recoverable: surfaced as a tool observation so the agent can retry."""


class SearchBackendError(RuntimeError):
    """The search backend is unreachable or persistently failing (5xx).
    Unrecoverable for this rollout; the caller should abort the sample
    rather than silently reward 0."""


# Cap concurrent requests to the search server across all rollouts in this
# process: bursts of parallel rollouts otherwise overflow its request queue
# and it starts returning 408s. One semaphore per event loop — train and
# eval rollouts may run in different loops (e.g. the fully-async worker
# thread vs the shared async-utils loop), and an asyncio.Semaphore binds to
# the loop that first awaits it.
_SEARCH_SEMAPHORES: dict = {}


def _search_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _SEARCH_SEMAPHORES.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(int(os.getenv("BROWSECOMP_SEARCH_CONCURRENCY", "16")))
        _SEARCH_SEMAPHORES[loop] = sem
    return sem


class AsyncSearchClient:
    def __init__(self, base_url: str, timeout: float = 300.0, retries: int = 3, backoff: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self._client = httpx.AsyncClient(base_url=self.base_url)

    async def close(self):
        await self._client.aclose()

    async def _post(self, path: str, payload: dict):
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                async with _search_semaphore():
                    r = await self._client.post(path, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("results", data)
            except httpx.ConnectError as e:
                raise SearchBackendError(
                    f"Search backend at {self.base_url} is unreachable ({e}). "
                    f"Check LOCAL_SEARCH_URL and that the search server is running."
                ) from e
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (400, 422):
                    # FastAPI rejected the model-generated query/URL (e.g.
                    # whitespace-only search query, missing docid/url on
                    # /open). Recoverable via a corrected request.
                    body = e.response.text[:500]
                    raise SearchRequestError(f"{path} rejected request with HTTP {status}: {body}") from e
                if status == 408:
                    # The server's internal queue timed out under load.
                    # Back off harder and retry; if it persists, surface as
                    # a recoverable observation instead of killing the
                    # rollout — the agent can reissue the search.
                    last_exc = e
                    if attempt == self.retries:
                        raise SearchRequestError(
                            f"{path} timed out under load (HTTP 408) after {self.retries} attempts; "
                            f"the search backend is busy, try again"
                        ) from e
                    await asyncio.sleep(2.0 * attempt)
                    continue
                last_exc = e
                if attempt == self.retries:
                    raise SearchBackendError(
                        f"Search backend at {self.base_url}{path} failed after "
                        f"{self.retries} attempts with HTTP {status}: {e}"
                    ) from e
                await asyncio.sleep(self.backoff * attempt)
            except httpx.HTTPError as e:
                last_exc = e
                if attempt == self.retries:
                    raise SearchBackendError(
                        f"Search backend at {self.base_url}{path} failed after {self.retries} attempts: {e}"
                    ) from e
                await asyncio.sleep(self.backoff * attempt)
        raise last_exc  # should not reach

    async def search(self, query: str, k: int = 10):
        return await self._post("/search", {"query": query, "k": k})

    async def open(self, url: str | None = None, docid: str | None = None):
        return await self._post("/open", {"url": url, "docid": docid})


def extract_json_tool(text: str):
    """Return [{"function": ..., "arguments": {...}}, ...] from <tool_call>
    and <answer> blocks; ignore others."""
    calls = []

    def parse_obj(s):
        for p in (json.loads, ast.literal_eval):
            try:
                return p(s)
            except Exception:
                pass
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            frag = m.group(0)
            for p in (json.loads, ast.literal_eval):
                try:
                    return p(frag)
                except Exception:
                    pass
        return None

    for kind, body in re.findall(r"<(tool_call|answer)>\s*(.*?)\s*</\1>", text, flags=re.S):
        body = body.strip()
        if kind == "tool_call":
            if body.startswith("```") and body.endswith("```"):
                body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.S).strip()
            obj = parse_obj(body)
            if isinstance(obj, dict) and "name" in obj:
                args = obj.get("arguments", {})
                calls.append({"function": obj["name"], "arguments": args if isinstance(args, dict) else {}})
        elif kind == "answer":
            calls.append({"function": "finish", "arguments": {"answer": body}})

    aligned_calls = []
    for fn in calls:
        if fn["function"] == "search":
            topk = max(10 // (len(fn["arguments"].get("query", [])) + 1), 2)
            for q in fn["arguments"].get("query", []):
                aligned_calls.append({"function": "search", "arguments": {"query": q, "topk": topk}})
        elif fn["function"] == "visit":
            for url in fn["arguments"].get("url", []):
                aligned_calls.append({"function": "open_page", "arguments": {"url": url}})
        else:
            aligned_calls.append(fn)
    return aligned_calls


def extract_fn_call(text):
    """Parse the last group of `<function=name>...<parameter=key>value</parameter>...</function>`
    calls (the format specified in the BrowseComp system prompt), with a
    fallback for `<tool_call>` JSON / `<answer>` blocks."""
    if not text:
        return None
    if "<tool_call>" in text or "<answer>" in text:
        json_tool = extract_json_tool(text)
        if len(json_tool) > 0:
            return json_tool
    text = re.split(r"<\[[^\]]+\]>", text)[-1].strip()
    matches = list(re.finditer(r"(?m)^[ \t]*<function=([^>]+)>\s*(.*?)\s*</function>", text, re.DOTALL))
    if not matches:
        return None
    groups = [[matches[0]]]
    for m in matches[1:]:
        prev = groups[-1][-1]
        line_gap = text.count("\n", prev.end(), m.start())
        groups[-1].append(m) if line_gap < 4 else groups.append([m])
    last = groups[-1]
    return [
        {
            "function": m.group(1),
            "arguments": dict(re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", m.group(2), re.DOTALL)),
        }
        for m in last
    ]


REFLECT_NUDGE = (
    "\n\n* Please reflect on the information we have obtained, and keep searching for "
    "additional information if we still can not answer the question. Do not give the "
    "answer if the information is still not enough."
)

RESUBMIT_MESSAGE = (
    "Fail to parse answer. Please resubmit with the correct tool call format, eg\n"
    "<function=finish>\n"
    "<parameter=answer>YOUR ANSWER</parameter>\n"
    "<parameter=explanation>YOUR EXPLANATION</parameter>\n"
    "<parameter=confidence>YOUR CONFIDENCE</parameter>\n"
    "</function>\n"
)

MUST_SEARCH_MESSAGE = (
    "Answer submission failed. You MUST use the search tool to verify the answer and "
    "all the evidence, and cite the correct source document in your explanation to "
    "support your claim."
)


class BrowseCompEnv:
    """Per-rollout tool-execution state machine (search / open_page / finish).

    Ported from FoldAgent's LocalSearch env. `run_action` executes the function
    calls found in one assistant response and returns either
    {"observation": str} (continue the loop) or {"action": "finish"} (done,
    with the submitted answer in `self.predicted_answer`).
    """

    def __init__(self, question: str, label_answer: str, must_search: bool = True):
        base_url = os.getenv("LOCAL_SEARCH_URL")
        assert base_url, "LOCAL_SEARCH_URL must point at the BrowseComp-Plus search server"
        self.client = AsyncSearchClient(base_url=base_url)
        self.question = question
        self.label_answer = label_answer
        self.predicted_answer: tuple | None = None  # (answer, explanation, confidence)
        self.must_search = must_search
        # Off by default, matching FoldAgent's trained configuration.
        self.donotgiveup = os.getenv("BROWSECOMP_DO_NOT_GIVE_UP", "0") == "1"
        self.visited_pages: set = set()
        self.is_finish = False
        self.stats = Counter({"search": 0, "open_page": 0, "finish": 0, "change_answer": 0})

    async def close(self):
        await self.client.close()

    async def run_action(self, response: str) -> dict:
        fn_call = extract_fn_call(response)
        if not fn_call:
            return {"observation": "No function call was detected in the model response."}

        observation = ""
        for fn in fn_call:
            name = fn["function"]
            if name == "search":
                self.stats["search"] += 1
                query = fn["arguments"].get("query", "")
                topk = (lambda v: int(v) if str(v).isdigit() else 10)(fn["arguments"].get("topk", 10))
                topk = min(topk, _int_env("BROWSECOMP_SEARCH_MAX_TOPK", 10))
                if not query:
                    observation += '[Error] The "search" function requires a "query" argument.'
                    continue
                observation += f'[Search Results for "{query}"]\n'
                try:
                    serp = await self.client.search(query, 50)
                except SearchRequestError as e:
                    observation += f"[Error] search backend rejected the query: {e}\n"
                    continue
                show_topk = 0
                for i, page in enumerate(serp, 1):
                    if page["docid"] in self.visited_pages:
                        page["text"] = (
                            "(This page was already seen in a previous search. Here, a shorter "
                            "snippet is shown. If you find this page relevant, please use the "
                            "open_page tool to inspect the full content) "
                            + " ".join(page["text"].split()[:128])
                        )
                        show_topk += 0.25
                    else:
                        self.visited_pages.add(page["docid"])
                        page["text"] = " ".join(page["text"].split()[: _int_env("BROWSECOMP_SEARCH_SNIPPET_WORDS", 512)])
                        show_topk += 1
                    observation += (
                        f"\n--- #{i}: {page['docid']}---\n"
                        f"docid: {page['docid']}\n"
                        f"url: {page['url']}\n"
                        f"content: {page['text']}\n"
                    )
                    if show_topk >= topk:
                        break
                observation += "\n"

            elif name == "open_page":
                self.stats["open_page"] += 1
                url = fn["arguments"].get("url", None)
                docid = fn["arguments"].get("docid", None)
                if not docid and not url:
                    observation += '[Error] The "open_page" function requires either a "docid" or a "url".'
                    continue
                try:
                    open_pages = await self.client.open(url, docid)
                except SearchRequestError as e:
                    observation += f"[Error] open_page backend rejected the request: {e}\n"
                    continue
                for page in open_pages:
                    page["text"] = keep_first_n_words(page["text"], _int_env("BROWSECOMP_OPEN_PAGE_WORDS", 4096))
                    observation += (
                        f"[Opened Page Content]\n"
                        f"docid: {page['docid']}\n"
                        f"url: {page['url']}\n"
                        f"content: {page['text']}\n"
                    )
                observation += "\n"

            elif name == "finish":
                self.stats["finish"] += 1
                answer = fn["arguments"].get("answer", "")
                explanation = fn["arguments"].get("explanation", None)
                confidence = fn["arguments"].get("confidence", None)
                if len(answer.strip()) == 0:
                    return {"observation": RESUBMIT_MESSAGE.strip()}
                if self.predicted_answer is not None and self.predicted_answer[0] != answer:
                    self.stats["change_answer"] += 1

                if self.stats["search"] == 0:
                    # Answering from parametric memory alone: force at least one
                    # search when the guess is already right, and never reward
                    # a search-free answer.
                    if em_score(self.label_answer, answer) and self.must_search:
                        self.must_search = False
                        return {"observation": MUST_SEARCH_MESSAGE}
                    answer = ""  # no search, no reward
                self.predicted_answer = (answer, explanation, confidence)

                if "insufficient" in answer.lower() and self.donotgiveup:
                    self.donotgiveup = False
                    return {
                        "observation": (
                            "The answer is guaranteed to be found through sufficient search and "
                            "reading. Do not give up; try searching deeper or using alternative "
                            "approaches."
                        )
                    }

                if "<q1>" in self.label_answer:
                    label_answer_dict = extract_q_dict(self.label_answer)
                    predicted_answer_dict = extract_q_dict(self.predicted_answer[0])
                    missing = [k for k in label_answer_dict if k not in predicted_answer_dict]
                    if missing:
                        return {
                            "observation": (
                                f"Answer submission failed. The answer is missing the following "
                                f"questions: {', '.join(missing)}. Make sure submit answer for all "
                                f"the questions. Ensure all the answers are submitted in one finish "
                                f"tool call."
                            )
                        }

                self.is_finish = True
                return {"action": "finish"}
            else:
                observation = f'[Error] The function "{name}" is not supported.'

        observation += REFLECT_NUDGE
        return {"observation": observation.strip()}
