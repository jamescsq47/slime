#!/usr/bin/env python3
"""Small request-sticky HTTP proxy for independent DualPD engine pairs.

Every generation of one agent carries the same ``agentic_request_id``.  Hashing
that immutable ID keeps the complete request-generation lifecycle inside one
P<->D control group while allowing several TP groups to share one workload
endpoint.  The proxy makes no KV, path, admission, or retry decisions.
"""

import argparse
import hashlib
import json

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


def request_owner(payload: dict) -> str:
    custom = payload.get("custom_params")
    if isinstance(custom, dict):
        value = custom.get("agentic_request_id")
        if value:
            return str(value)
    # The harness mirrors its lifecycle envelope here for routers that strip
    # backend extensions.  It begins with a generation-independent encoded ID.
    user = payload.get("user")
    if isinstance(user, str) and user:
        return user.split(":g", 1)[0]
    extra = payload.get("extra_key")
    if isinstance(extra, str) and extra:
        return extra.split(":g", 1)[0]
    raise web.HTTPBadRequest(text="agentic_request_id is required")


def owner_index(owner: str, count: int) -> int:
    digest = hashlib.blake2b(owner.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % count


async def create_app(upstreams: list[str]) -> web.Application:
    timeout = ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=None)
    session = ClientSession(timeout=timeout, connector=TCPConnector(limit=0))
    app = web.Application(client_max_size=64 * 1024**2)
    app["session"] = session
    app["upstreams"] = [item.rstrip("/") for item in upstreams]

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "upstreams": len(upstreams)})

    async def forward(request: web.Request) -> web.Response:
        raw = await request.read()
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise web.HTTPBadRequest(text="request body must be JSON") from exc
        owner = request_owner(payload)
        index = owner_index(owner, len(upstreams))
        target = app["upstreams"][index] + request.rel_url.path_qs
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        async with session.request(request.method, target, data=raw, headers=headers) as response:
            body = await response.read()
            output_headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower() not in {"content-length", "connection", "transfer-encoding"}
            }
            output_headers["x-dualpd-pair"] = str(index)
            return web.Response(status=response.status, body=body, headers=output_headers)

    async def close_session(_app: web.Application) -> None:
        await session.close()

    app.router.add_get("/health", health)
    app.router.add_route("*", "/{tail:.*}", forward)
    app.on_cleanup.append(close_session)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", action="append", required=True)
    args = parser.parse_args()
    web.run_app(create_app(args.upstream), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
