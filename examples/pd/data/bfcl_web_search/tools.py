"""Live, keyless tools. No fake latency, result cache, gold data, or local corpus."""

import asyncio
import ipaddress
import json
import socket
import time
from urllib.parse import urljoin, urlsplit
from weakref import WeakKeyDictionary

import aiohttp
from bs4 import BeautifulSoup


class WebBackendError(RuntimeError):
    pass


class PublicResolver(aiohttp.resolver.DefaultResolver):
    """Validate the addresses actually supplied to the socket connector."""

    async def resolve(self, host, port=0, family=socket.AF_INET):
        records = await super().resolve(host, port, family)
        if not records or any(not ipaddress.ip_address(r["host"]).is_global for r in records):
            raise ValueError("Private/local network addresses are not allowed")
        return records


_limits = WeakKeyDictionary()


def io_limit(kind):
    loop = asyncio.get_running_loop()
    limits = _limits.setdefault(loop, {})
    return limits.setdefault(kind, asyncio.Semaphore(4 if kind == "search" else 8))


async def public_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.port not in {None, 80, 443}):
        raise ValueError("Only public HTTP(S) URLs without credentials are allowed")
    addresses = await asyncio.get_running_loop().getaddrinfo(
        parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
        type=socket.SOCK_STREAM,
    )
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("Private/local network addresses are not allowed")


def validate_call(call):
    if not isinstance(call, dict) or not isinstance(call.get("arguments", {}), dict):
        raise ValueError("Tool call and arguments must be JSON objects")
    name, args = call.get("name"), dict(call.get("arguments", {}))
    if name == "search_engine_query":
        if set(args) - {"keywords", "max_results", "region"}:
            raise ValueError("Unknown search argument")
        if not isinstance(args.get("keywords"), str) or not args["keywords"].strip():
            raise ValueError("keywords must be a nonempty string")
        if len(args["keywords"]) > 2000:
            raise ValueError("Search query exceeds 2000 characters")
        count = args.get("max_results", 10)
        if type(count) is not int or not 1 <= count <= 10:
            raise ValueError("max_results must be an integer from 1 to 10")
        if not isinstance(args.get("region", "wt-wt"), str):
            raise ValueError("region must be a string")
    elif name == "fetch_url_content":
        if set(args) - {"url", "mode"}:
            raise ValueError("Unknown fetch argument")
        if not isinstance(args.get("url"), str):
            raise ValueError("url must be a string")
        if args.get("mode", "raw") not in {"raw", "markdown", "truncate"}:
            raise ValueError("Unknown page mode")
    else:
        raise ValueError(f"Unknown tool: {name}")
    return name, args


class WebTools:
    def __init__(self, *, backend="auto", snippets=True, timeout=20, max_chars=40000):
        self.backend, self.snippets = backend, snippets
        self.timeout, self.max_chars = timeout, max_chars
        self.events = []

    async def _search(self, args):
        from ddgs import DDGS
        from ddgs.engines import ENGINES
        # DDGS silently falls back to auto for disabled/misspelled providers.
        # Reject that case instead of recording a provider we did not use.
        if self.backend != "auto" and self.backend not in ENGINES["text"]:
            raise ValueError(f"Disabled/unknown search backend: {self.backend}")
        # BFCL/SerpAPI uses wt-wt for unspecified region. DDGS treats its last
        # component as a language, which would request nonexistent wt.wikipedia.
        region = args.get("region", "wt-wt")
        region = "us-en" if region == "wt-wt" else region
        def run():
            return DDGS(timeout=self.timeout).text(
                args["keywords"], region=region,
                max_results=args.get("max_results", 10), backend=self.backend,
            )
        # Cancellation must not release the concurrency slot while the actual
        # blocking web search is still running in its thread.
        gate = io_limit("search")
        await gate.acquire()
        future = asyncio.create_task(asyncio.to_thread(run))
        future.add_done_callback(lambda f: (gate.release(), None if f.cancelled() else f.exception()))
        rows = await asyncio.shield(future)
        if not rows:
            raise WebBackendError("Search backend returned no results")
        fields = ("title", "href", "body") if self.snippets else ("title", "href")
        return [{k: str(row.get(k, "")) for k in fields} for row in rows]

    async def _fetch(self, args):
        url = args["url"]
        async with io_limit("fetch"), aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(resolver=PublicResolver()),
            timeout=aiohttp.ClientTimeout(total=self.timeout), trust_env=False,
            headers={"User-Agent": "BFCL-serving-research/1.0"},
        ) as client:
            for _ in range(6):
                await public_url(url)
                async with client.get(url, allow_redirects=False) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        url = urljoin(url, response.headers["location"])
                        continue
                    response.raise_for_status()
                    chunks, size = [], 0
                    async for chunk in response.content.iter_chunked(65536):
                        size += len(chunk)
                        if size > 2_000_000:
                            raise WebBackendError("Page exceeds the 2 MB download limit")
                        chunks.append(chunk)
                    text = b"".join(chunks).decode(response.charset or "utf-8", errors="replace")
                    break
            else:
                raise WebBackendError("Too many redirects")
        mode = args.get("mode", "raw")
        if mode == "markdown":
            import html2text
            text = await asyncio.to_thread(html2text.html2text, text)
        elif mode == "truncate":
            def clean():
                soup = BeautifulSoup(text, "html.parser")
                for node in soup(["script", "style", "noscript"]):
                    node.decompose()
                return " ".join(soup.stripped_strings)
            text = await asyncio.to_thread(clean)
        return {"content": text[:self.max_chars], "truncated": len(text) > self.max_chars}

    async def call(self, call):
        name, args = validate_call(call)
        event = {"tool": name, "arguments": args, "backend": self.backend, "ok": False}
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout * 3):
                result = await (self._search(args) if name == "search_engine_query" else self._fetch(args))
            event["ok"] = True
            event["output_chars"] = len(json.dumps(result, ensure_ascii=False))
            return result
        except asyncio.CancelledError:
            event["error"] = "cancelled"
            raise
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            raise WebBackendError(event["error"]) from exc
        finally:
            event["elapsed_seconds"] = time.monotonic() - started
            self.events.append(event)
