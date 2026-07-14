"""
Local Search Server for Search-R1

This module provides a local search engine interface that mimics the google_search_server.py API.
It sends requests to a local retrieval server (e.g., running retrieval_server.py from Search-R1)
and formats the results to match the expected output format.

Usage:
    In your generate_with_search.py, replace:
        from google_search_server import google_search
    with:
        from local_search_server import local_search as google_search

    And update SEARCH_R1_CONFIGS:
        SEARCH_R1_CONFIGS = {
            "search_url": "http://127.0.0.1:8000/retrieve",  # URL of local retrieval server
            "topk": 3,
            ...
        }
"""

import asyncio
import weakref

import aiohttp

_LOOP_SESSIONS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, aiohttp.ClientSession] = weakref.WeakKeyDictionary()
_LOOP_SEMAPHORES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = weakref.WeakKeyDictionary()
_MAX_CONCURRENT_SEARCH_REQUESTS = 128


def _get_loop_search_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _LOOP_SEMAPHORES.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SEARCH_REQUESTS)
        _LOOP_SEMAPHORES[loop] = semaphore
    return semaphore


async def close_loop_sessions() -> None:
    loop = asyncio.get_running_loop()
    session = _LOOP_SESSIONS.pop(loop, None)
    _LOOP_SEMAPHORES.pop(loop, None)
    if session is not None and not session.closed:
        await session.close()


def _get_loop_session() -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    session = _LOOP_SESSIONS.get(loop)
    if session is None or session.closed:
        connector = aiohttp.TCPConnector(
            limit=_MAX_CONCURRENT_SEARCH_REQUESTS,
            limit_per_host=_MAX_CONCURRENT_SEARCH_REQUESTS,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )
        session = aiohttp.ClientSession(connector=connector)
        _LOOP_SESSIONS[loop] = session
    return session


async def local_search(
    search_url: str,
    query: str,
    top_k: int = 5,
    timeout: int = 60,
    proxy: str | None = None,
) -> list[dict]:
    """
    Call local search engine server and format results to match google_search_server.py output.

    This function provides the same interface as google_search() from google_search_server.py,
    making it a drop-in replacement. The only difference is that instead of using an API key,
    it uses a search_url parameter.

    Args:
        search_url: URL of the local retrieval server (e.g., "http://127.0.0.1:8000/retrieve")
        query: Search query string
        top_k: Number of results to retrieve
        timeout: Request timeout in seconds (default: 60)
        proxy: Proxy URL if needed (not used for local retrieval, kept for API compatibility)
        snippet_only: If True, only return snippet (kept for API compatibility with google_search)

    Returns:
        List of dictionaries with format: [{"document": {"contents": '"<title>"\n<text>'}}]
        This matches the output format of google_search() from google_search_server.py
    """
    # Prepare request payload for local retrieval server
    payload = {
        "queries": [query],
        "topk": top_k,
        "return_scores": False,  # We don't need scores for compatibility with google_search_server
    }

    # Send async request to local retrieval server. Keep one session and one
    # concurrency gate per event loop: this avoids closing a connector shared by
    # other tasks, avoids per-request connector churn, and prevents rollout from
    # flooding the retriever with unbounded simultaneous HTTP requests.
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    result = None
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            kwargs = {}
            if proxy:
                kwargs["proxy"] = proxy
            async with _get_loop_search_semaphore():
                session = _get_loop_session()
                async with session.post(search_url, json=payload, timeout=timeout_obj, **kwargs) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
            break
        except Exception as e:
            should_retry = attempt < max_attempts
            print(
                f"Error calling local search engine at {search_url} "
                f"(attempt {attempt}/{max_attempts}, query={query[:120]!r}): "
                f"{type(e).__name__}: {e!r}",
                flush=True,
            )
            if not should_retry:
                return []
            await asyncio.sleep(0.2 * attempt)

    if result is None:
        return []

    # Parse retrieval results
    # Format from retrieval_server.py: {"result": [[{"document": {"id": "...", "contents": "..."}}]]}
    retrieval_results = result.get("result", [[]])[0]
    # Format to match google_search_server.py output
    # Google format: [{"document": {"contents": '"<title>"\n<context>'}}]
    contexts = []

    for item in retrieval_results:
        # Extract contents from retrieval result
        # retrieval_server returns: {"document": {"id": "...", "contents": '"Title"\nText...'}}
        if isinstance(item, dict):
            # Access the document dict first, then get contents
            content = item.get("contents", "")

            if content:
                # The contents are already in the correct format: '"Title"\nText content...'
                # Just pass through as-is to match google_search format
                contexts.append({"document": {"contents": content}})
            else:
                # Empty content case - provide default values
                contexts.append({"document": {"contents": '"No title."\nNo snippet available.'}})

    # If no results found, return empty list (consistent with google_search_server.py)
    return contexts
