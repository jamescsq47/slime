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


class TerminalEnvironmentError(RuntimeError):
    """Structured OpenEnv error returned over the WebSocket protocol."""

    def __init__(self, message: str, code: str):
        self.code = str(code)
        super().__init__(
            f"Terminal-Bench environment error: {message} (code={self.code})"
        )


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
        try:
            await self._socket.send(json.dumps(message))
            raw = await asyncio.wait_for(
                self._socket.recv(), timeout=self.message_timeout
            )
        except Exception as exc:
            # The current OpenEnv server closes some reset connections with a
            # normal WebSocket close (1000) when saturated instead of sending
            # its structured CAPACITY_REACHED response.  Preserve that fact
            # as a structured code so the harness can retry *only* while
            # acquiring an environment; a close during an active task still
            # remains a real sample failure.
            try:
                from websockets.exceptions import ConnectionClosedOK
            except ImportError:
                ConnectionClosedOK = ()
            if isinstance(exc, ConnectionClosedOK):
                raise TerminalEnvironmentError(
                    str(exc), "RESET_CONNECTION_CLOSED_OK"
                ) from exc
            raise
        response = json.loads(raw)
        if response.get("type") == "error":
            data = response.get("data") or {}
            raise TerminalEnvironmentError(
                str(data.get("message", "unknown")),
                str(data.get("code", "UNKNOWN")),
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
        socket = self._socket
        self._socket = None

        async def close_remote() -> None:
            from websockets.exceptions import ConnectionClosed

            try:
                await socket.send(json.dumps({"type": "close"}))
                # The server removes the task container while handling this
                # request.  Wait for its reply before closing the WebSocket;
                # otherwise cancellation of a large rollout can strand many
                # live containers.
                await asyncio.wait_for(socket.recv(), timeout=60.0)
            except (ConnectionClosed, TimeoutError):
                # OpenEnv may close before or after receiving our close action,
                # and saturated cleanup may exceed this best-effort timeout.
                # Neither case changes the already-computed sample result.
                pass
            finally:
                await socket.close()

        cleanup = asyncio.create_task(close_remote())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # Rollout workers are cancelled together at the measurement
            # boundary.  Finish remote cleanup before propagating cancellation.
            await cleanup
            raise
