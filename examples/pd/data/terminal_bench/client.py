"""Minimal OpenEnv WebSocket client used by the Terminal-Bench harness."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse


@dataclass(frozen=True)
class EnvironmentResult:
    instruction: str = ""
    output: str = ""
    info: dict[str, Any] | None = None
    reward: float | None = None
    done: bool = False


class Tbench2Client:
    def __init__(self, base_url: str, *, message_timeout: float = 4200.0):
        parsed = urlparse(base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        self.ws_url = urlunparse(
            parsed._replace(scheme=scheme, path="/ws", params="", query="", fragment="")
        )
        self.message_timeout = message_timeout
        self._socket = None

    async def connect(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "Terminal-Bench requires websockets; install examples/pd/requirement.txt"
            ) from exc
        self._socket = await websockets.connect(
            self.ws_url,
            open_timeout=30.0,
            max_size=100 * 1024 * 1024,
        )

    async def _request(self, message: dict[str, Any]) -> EnvironmentResult:
        if self._socket is None:
            raise RuntimeError("Terminal-Bench environment is not connected")
        await self._socket.send(json.dumps(message))
        raw = await asyncio.wait_for(self._socket.recv(), timeout=self.message_timeout)
        response = json.loads(raw)
        if response.get("type") == "error":
            data = response.get("data") or {}
            raise RuntimeError(
                f"Terminal-Bench environment error: {data.get('message', 'unknown')} "
                f"(code={data.get('code', 'UNKNOWN')})"
            )
        data = response.get("data") or {}
        observation = data.get("observation") or {}
        return EnvironmentResult(
            instruction=str(observation.get("instruction") or ""),
            output=str(observation.get("output") or observation.get("error") or ""),
            info=dict(observation.get("info") or {}),
            reward=data.get("reward"),
            done=bool(data.get("done", False)),
        )

    async def reset(self, task_id: str) -> EnvironmentResult:
        return await self._request({"type": "reset", "data": {"task_id": task_id}})

    async def execute(self, command: str) -> EnvironmentResult:
        return await self._request(
            {
                "type": "step",
                "data": {
                    "action_type": "exec",
                    "command": command,
                    "session_id": None,
                    "block": True,
                    "wait_seconds": None,
                    "file_path": "",
                    "content": "",
                },
            }
        )

    async def evaluate(self) -> EnvironmentResult:
        return await self._request(
            {
                "type": "step",
                "data": {
                    "action_type": "evaluate",
                    "command": "",
                    "session_id": None,
                    "block": True,
                    "wait_seconds": None,
                    "file_path": "",
                    "content": "",
                },
            }
        )

    async def close(self) -> None:
        if self._socket is None:
            return
        try:
            await self._socket.send(json.dumps({"type": "close"}))
        finally:
            await self._socket.close()
            self._socket = None
