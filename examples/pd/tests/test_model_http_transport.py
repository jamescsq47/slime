import asyncio
from contextlib import asynccontextmanager
import gzip
from pathlib import Path
import sys
from types import SimpleNamespace

import aiohttp.web
import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_http_transport import AiohttpTransport, http_client_scope, init_http_client, transport_config
from slime.utils import http_utils


@asynccontextmanager
async def server(handler):
    app = aiohttp.web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def client(concurrency=8, **timeouts):
    return httpx.AsyncClient(
        transport=AiohttpTransport(concurrency), trust_env=False,
        timeout=httpx.Timeout(**({"connect": 10, "read": 10, "write": 10, "pool": 5} | timeouts)),
    )


def test_identical_wire_payload_headers_cookies_and_response():
    async def run():
        records = []
        async def handler(request):
            records.append((request.raw_path, dict(request.headers), await request.read()))
            response = aiohttp.web.Response(
                body=gzip.compress('{"text":"你好<think>reasoning</think>","tool_calls":[]}'.encode()),
                headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
            )
            response.set_cookie("session", "same")
            return response
        payload = {"input_ids": [1, 2, 3], "messages": [{"content": "特殊字符\n\\\"☺"}],
                   "extra_key": "stable-generation", "temperature": 0.6}
        async with server(handler) as url:
            async with httpx.AsyncClient(trust_env=False) as original, client() as replacement:
                results = []
                for c in (original, replacement):
                    for _ in range(2):
                        response = await c.post(url + "/generate?q=a%2Fb", json=payload, headers={"X-Request-ID": "same"})
                        results.append(response.json())
                assert records[:2] == records[2:]
                assert all(result == results[0] for result in results)
                transport = replacement._transport
                assert transport._session._retry_connection is False
            assert transport._session.closed
    asyncio.run(run())


def test_shared_retry_loop_status_text_redirect_and_identity():
    async def run():
        calls = []
        async def handler(request):
            calls.append(await request.read())
            if len(calls) == 1:
                return aiohttp.web.Response(status=503, text="retry body")
            if request.path == "/redirect":
                return aiohttp.web.Response(status=307, headers={"Location": "/unexpected"})
            return aiohttp.web.Response(text="plain response")
        async with server(handler) as url, client() as c:
            assert await http_utils._post(c, url, {"extra_key": "same"}, max_retries=2) == "plain response"
            assert len(calls) == 2 and calls[0] == calls[1]
            with pytest.raises(httpx.HTTPStatusError) as error:
                await http_utils._post(c, url + "/redirect", {}, max_retries=1)
            assert error.value.response.status_code == 307
            assert len(calls) == 3  # no redirect resend
    asyncio.run(run())


def test_read_timeout_and_cancel_release_pool_without_resend():
    async def run():
        count = 0
        entered = asyncio.Event()
        release = asyncio.Event()
        async def handler(request):
            nonlocal count
            count += 1
            entered.set()
            if request.path == "/hang":
                await release.wait()
            return aiohttp.web.json_response({"ok": True})
        async with server(handler) as url:
            try:
                async with client(1, read=0.05) as c:
                    with pytest.raises(httpx.ReadTimeout):
                        await http_utils._post(c, url + "/hang", {}, max_retries=1)
                    assert count == 1
                    assert (await c.post(url, json={})).json() == {"ok": True}
                    entered.clear()
                    task = asyncio.create_task(http_utils._post(c, url + "/hang", {}))
                    await entered.wait()
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert count == 3
                    assert (await c.get(url)).status_code == 200
                    assert c._transport._slots._value == 1
            finally:
                release.set()
    asyncio.run(run())


def test_pool_timeout_and_queued_cancel():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        count = 0
        async def handler(request):
            nonlocal count
            count += 1
            entered.set()
            await release.wait()
            return aiohttp.web.json_response({})
        async with server(handler) as url, client(1, pool=0.03) as c:
            first = asyncio.create_task(c.post(url, json={}))
            try:
                await entered.wait()
                with pytest.raises(httpx.PoolTimeout):
                    await c.post(url, json={})
                queued = asyncio.create_task(c.post(url, json={}))
                await asyncio.sleep(0)
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await queued
                assert count == 1
            finally:
                release.set()
                await first
            assert c._transport._slots._value == 1
    asyncio.run(run())


def test_disconnect_no_hidden_post_retry_and_status_body():
    async def run():
        count = 0
        async def handler(request):
            nonlocal count
            await request.read()
            count += 1
            if request.path == "/disconnect":
                request.transport.close()
                return aiohttp.web.Response()
            return aiohttp.web.Response(status=400, text="invalid request identity")
        async with server(handler) as url, client() as c:
            with pytest.raises(httpx.RemoteProtocolError):
                await http_utils._post(c, url + "/disconnect", {"id": "same"}, max_retries=1)
            assert count == 1
            with pytest.raises(httpx.HTTPStatusError) as error:
                await http_utils._post(c, url, {}, max_retries=1)
            assert error.value.response.text == "invalid request identity"
            assert count == 2
    asyncio.run(run())


def test_write_timeout_actual_blocked_socket():
    async def run():
        release = asyncio.Event()
        async def handler(reader, writer):
            try:
                await release.wait()  # deliberately never read request body
            finally:
                writer.close()
                await writer.wait_closed()
        listener = await asyncio.start_server(handler, "127.0.0.1", 0)
        url = f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        try:
            async with client(1, write=0.02, read=1) as c:
                with pytest.raises(httpx.WriteTimeout):
                    await c.post(url, content=b"x" * (32 * 1024 * 1024))
                assert c._transport._slots._value == 1
        finally:
            release.set()
            listener.close()
            await listener.wait_closed()
            await asyncio.sleep(0.02)
    asyncio.run(run())


def test_connection_refused():
    async def run():
        listener = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        port = listener.sockets[0].getsockname()[1]
        listener.close()
        await listener.wait_closed()
        async with client() as c:
            with pytest.raises(httpx.ConnectError):
                await c.get(f"http://127.0.0.1:{port}")
    asyncio.run(run())


def test_fragmented_response_and_truncated_payload():
    async def run():
        async def handler(request):
            if request.path == "/truncated":
                response = aiohttp.web.StreamResponse(headers={"Content-Length": "100"})
                await response.prepare(request)
                await response.write(b"partial")
                request.transport.close()
                return response
            response = aiohttp.web.StreamResponse()
            await response.prepare(request)
            for chunk in (b'{"tool', b'_calls":', b'[],"ok":true}'):
                await response.write(chunk)
                await asyncio.sleep(0.01)
            await response.write_eof()
            return response
        async with server(handler) as url, client(1, read=0.05) as c:
            assert await http_utils._post(c, url, {}, max_retries=1) == {"tool_calls": [], "ok": True}
            with pytest.raises(httpx.ReadError):
                await http_utils._post(c, url + "/truncated", {}, max_retries=1)
            assert c._transport._slots._value == 1
            assert (await c.get(url)).status_code == 200
    asyncio.run(run())


def test_connect_timeout_category_and_capacity_release(monkeypatch):
    async def connect_timeout(*args, **kwargs):
        raise aiohttp.ConnectionTimeoutError("injected connect timeout")
    monkeypatch.setattr(aiohttp.TCPConnector, "connect", connect_timeout)
    async def run():
        async with client(1) as c:
            with pytest.raises(httpx.ConnectTimeout):
                await c.post("http://127.0.0.1:1", json={})
            assert c._transport._slots._value == 1
    asyncio.run(run())


def test_scope_teardown_and_default_isolation(monkeypatch):
    args = SimpleNamespace(sglang_server_concurrency=500, rollout_num_gpus=2,
                           rollout_num_gpus_per_engine=1, use_distributed_post=False,
                           sglang_router_request_timeout_secs=3600)
    monkeypatch.delenv("PD_MODEL_HTTP_TRANSPORT", raising=False)
    assert transport_config(args)["backend"] == "httpx"
    monkeypatch.setenv("PD_MODEL_HTTP_TRANSPORT", "aiohttp")
    assert transport_config(args)["max_connections"] == 1000
    async def run():
        previous = http_utils._http_client
        with pytest.raises(RuntimeError, match="experiment failed"):
            async with http_client_scope():
                init_http_client(args)
                c = http_utils._http_client
                assert isinstance(c._transport, AiohttpTransport)
                with pytest.raises(RuntimeError, match="already active"):
                    init_http_client(args)
                raise RuntimeError("experiment failed")
        assert c.is_closed
        assert http_utils._http_client is previous
    asyncio.run(run())
    asyncio.run(run())  # no session tied to a previous event loop


def test_500_concurrent_requests_reuse_and_release():
    async def run():
        identities, sockets = [], set()
        async def handler(request):
            identities.append((await request.json())["id"])
            sockets.add(request.transport)
            return aiohttp.web.json_response({"ok": True})
        async with server(handler) as url, client(64) as c:
            for round_id in range(2):
                responses = await asyncio.gather(*[
                    http_utils._post(c, url, {"id": f"{round_id}:{i}"}, max_retries=1)
                    for i in range(500)
                ])
                assert all(r == {"ok": True} for r in responses)
            assert len(identities) == len(set(identities)) == 1000
            assert len(sockets) <= 64
            assert c._transport._slots._value == 64
            assert not c._transport._session.connector._acquired
    asyncio.run(run())
