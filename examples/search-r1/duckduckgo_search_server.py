import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor

from duckduckgo_search import DDGS


# 保留独立线程池，避免 fallback 到同步 DDGS 时挤占默认线程池。
DDG_EXECUTOR = ThreadPoolExecutor(max_workers=300)
MAX_RETRIES = 3


def _should_retry(error: Exception) -> bool:
    error_msg = str(error).lower()
    return any(token in error_msg for token in ("429", "rate", "timeout", "proxy"))


def _format_results(raw_results):
    formatted_results = []
    for res in raw_results:
        title = res.get("title", "No Title")
        snippet = res.get("body", "")
        contents = f"{title}\n{snippet}"
        formatted_results.append({"document": {"contents": contents}})
    return formatted_results


def _run_sync_duckduckgo_search(
    query: str,
    max_results: int,
    proxy: str = None,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    timeout: int = 15,
):
    for attempt in range(MAX_RETRIES):
        try:
            with DDGS(proxy=proxy, timeout=timeout) as ddgs:
                results = ddgs.text(
                    query,
                    max_results=max_results,
                    region=region,
                    safesearch=safesearch,
                )
                return list(results) if results else []
        except Exception as error:
            if _should_retry(error) and attempt < MAX_RETRIES - 1:
                sleep_time = random.uniform(1, 4) * (attempt + 1)
                print(
                    f"[DDG 限制] 查询 '{query}' 遇到限流/网络错误，等待 {sleep_time:.1f}s 后重试 "
                    f"({attempt + 1}/{MAX_RETRIES})..."
                )
                time.sleep(sleep_time)
                continue
            raise

    return []


async def _run_async_duckduckgo_search(
    query: str,
    max_results: int,
    proxy: str = None,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    timeout: int = 15,
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        DDG_EXECUTOR,
        _run_sync_duckduckgo_search,
        query,
        max_results,
        proxy,
        region,
        safesearch,
        timeout,
    )


async def duckduckgo_search(
    query: str,
    max_results: int,
    proxy: str = None,
    region: str = "wt-wt",
    safesearch: str = "moderate",
    timeout: int = 15,
):
    """
    DuckDuckGo 搜索封装。
    统一走 DDGS + 独立线程池，兼顾高并发和旧版本包环境。
    """
    try:
        raw_results = await _run_async_duckduckgo_search(
            query,
            max_results,
            proxy=proxy,
            region=region,
            safesearch=safesearch,
            timeout=timeout,
        )
    except Exception as error:
        print(f"[DuckDuckGo Search Error]: Query '{query}' failed with error: {error}")
        return []

    return _format_results(raw_results)