"""Opt-in experiment transport; HTTPX builds messages, aiohttp owns sockets.

The shared slime retry loop and all harnesses remain unchanged. In particular,
this transport never retries a request. The existing outer retry loop is NOT an
exactly-once guarantee after an ambiguous server/connection failure.
"""

import asyncio
from contextlib import asynccontextmanager
import os

import aiohttp
import httpx
from yarl import URL


class _WriteTimeout(TimeoutError):
    pass


class _TimedBody(aiohttp.payload.BytesPayload):
    def __init__(self, body, timeout):
        super().__init__(body)
        self.timeout = timeout

    async def write_with_length(self, writer, content_length):
        try:
            async with asyncio.timeout(self.timeout):
                await super().write_with_length(writer, content_length)
        except TimeoutError as exc:
            # aiohttp propagates an OSError/TimeoutError from body writing to
            # its reader. A distinct subtype preserves HTTPX's write category.
            raise _WriteTimeout("model HTTP body write timed out") from exc


class AiohttpTransport(httpx.AsyncBaseTransport):
    """Buffered JSON model calls only; no second pool scan or retry layer."""

    def __init__(self, concurrency):
        if concurrency <= 0:
            raise ValueError("model HTTP concurrency must be positive")
        self.concurrency = concurrency
        self._slots = asyncio.Semaphore(concurrency)
        self._session = None

    def _get_session(self):
        if self._session is None:
            session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    limit=self.concurrency, keepalive_timeout=30,
                ),
                cookie_jar=aiohttp.DummyCookieJar(),  # HTTPX owns cookies.
                auto_decompress=False,  # HTTPX decodes exactly once.
                trust_env=False,
                skip_auto_headers={"User-Agent", "Content-Type", "Accept-Encoding"},
            )
            # aiohttp retries a stale persistent GET once by default. Keep
            # *all* retry decisions in the existing slime get/_post loops.
            session._retry_connection = False
            self._session = session
        return self._session

    async def handle_async_request(self, request):
        timeout = request.extensions.get("timeout", {})
        try:
            async with asyncio.timeout(timeout.get("pool")):
                await self._slots.acquire()
        except TimeoutError as exc:
            raise httpx.PoolTimeout("model HTTP pool wait timed out", request=request) from exc
        try:
            body = await request.aread()
            session = self._get_session()
            async with session.request(
                request.method, URL(str(request.url), encoded=True),
                headers=[(k.decode("latin-1"), v.decode("latin-1")) for k, v in request.headers.raw],
                data=_TimedBody(body, timeout.get("write")) if body else None,
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(
                    total=None, connect=timeout.get("connect"),
                    sock_connect=timeout.get("connect"), sock_read=timeout.get("read"),
                    ceil_threshold=float("inf"),
                ),
            ) as response:
                content = await response.read()
                return httpx.Response(
                    response.status, headers=response.raw_headers,
                    stream=httpx.ByteStream(content),
                    extensions={"http_version": f"HTTP/{response.version.major}.{response.version.minor}".encode()},
                )
        except _WriteTimeout as exc:
            raise httpx.WriteTimeout(str(exc), request=request) from exc
        except aiohttp.ConnectionTimeoutError as exc:
            raise httpx.ConnectTimeout(str(exc), request=request) from exc
        except aiohttp.ServerTimeoutError as exc:
            raise httpx.ReadTimeout(str(exc), request=request) from exc
        except aiohttp.ClientConnectorError as exc:
            raise httpx.ConnectError(str(exc), request=request) from exc
        except aiohttp.ClientPayloadError as exc:
            raise httpx.ReadError(str(exc), request=request) from exc
        except aiohttp.ClientConnectionError as exc:
            raise httpx.RemoteProtocolError(str(exc), request=request) from exc
        finally:
            self._slots.release()

    async def aclose(self):
        if self._session is not None:
            await self._session.close()


def transport_config(args):
    backend = os.environ.get("PD_MODEL_HTTP_TRANSPORT", "httpx").lower()
    if backend not in {"httpx", "aiohttp"}:
        raise ValueError("PD_MODEL_HTTP_TRANSPORT must be httpx or aiohttp")
    return {
        "backend": backend,
        "max_connections": args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine,
        "keepalive_seconds": 30,
        "connect_timeout_seconds": 10,
        "read_timeout_seconds": getattr(args, "sglang_router_request_timeout_secs", 120),
        "write_timeout_seconds": 10,
        "pool_timeout_seconds": 5,
        "transport_retries": 0,
        "outer_retry_policy": "unchanged slime.utils.http_utils",
    }


def init_http_client(args):
    from slime.utils import http_utils

    config = transport_config(args)
    if config["backend"] == "httpx":
        return http_utils.init_http_client(args)
    if args.use_distributed_post or http_utils._distributed_post_enabled:
        raise ValueError("aiohttp experiment transport does not replace Ray actor clients")
    if not args.rollout_num_gpus:
        return
    if http_utils._http_client is not None:
        raise RuntimeError("refusing to replace an already active HTTP client")
    http_utils._client_concurrency = config["max_connections"]
    http_utils._http_client = httpx.AsyncClient(
        transport=AiohttpTransport(config["max_connections"]),
        timeout=httpx.Timeout(connect=10, read=config["read_timeout_seconds"], write=10, pool=5),
        trust_env=False,
    )


@asynccontextmanager
async def http_client_scope():
    """Close this experiment's client on success, cancellation or failure."""
    from slime.utils import http_utils

    previous = http_utils._http_client
    previous_concurrency = http_utils._client_concurrency
    try:
        yield
    finally:
        current = http_utils._http_client
        if current is not previous:
            try:
                if current is not None:
                    await current.aclose()
            finally:
                http_utils._http_client = previous
                http_utils._client_concurrency = previous_concurrency
