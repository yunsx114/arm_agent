"""NFS file-drop IPC: the CLIENT side (runs in the `qwen35` env, GPU node).

Mirror of sim/server.py. Publish a command with an atomic rename, then poll for
`resp_<id>.json`; image frames are read back as PIL images.

    client = SimClient(ipc_dir)
    client.reset(init_state_idx=0)          -> Response
    client.act({"kind": "move", ...})        -> Response
    client.close()

`Response.obs` is the packed observation dict; `Response.images` maps camera
name to a PIL image. Nothing here imports LIBERO/MuJoCo, so this module is safe
to import inside the agent process.
"""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _recv_exact(sock: Any, n: int) -> bytes:
    """Read exactly n bytes, or raise (a short read would desync the stream)."""
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise SimClientError(f"connection closed while expecting {n} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass
class Response:
    id: int
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def obs(self) -> dict[str, Any]:
        return self.result.get("obs", {})

    @property
    def outcome(self) -> dict[str, Any]:
        return self.result.get("outcome", {})

    def images(self, ipc_dir: Path | None = None) -> dict[str, Any]:
        """Camera frames of this response as PIL images.

        The TCP transport INLINES the PNGs as base64. The file-drop transport
        cannot be used there: NFS `acdirmin` defaults to 30 s, so a file the
        server just wrote stays invisible to a client that looks for it by
        name (measured: that is the 30.0 s stall per call seen in the episode
        timing breakdown). The on-disk path is kept for the in-process probes.
        """
        from PIL import Image

        inline = self.result.get("images_b64")
        if inline:
            return {
                cam: Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
                for cam, b in inline.items()
            }
        out = {}
        for cam, name in self.result.get("images", {}).items():
            out[cam] = Image.open(Path(ipc_dir) / name).convert("RGB")
        return out


class SimClientError(RuntimeError):
    pass


class SimClient:
    def __init__(
        self,
        ipc_dir: str | Path,
        timeout_s: float = 600.0,
        endpoint: tuple[str, int] | None = None,
    ) -> None:
        self.ipc_dir = Path(ipc_dir)
        self.timeout_s = timeout_s
        # endpoint=("host", port) selects TCP; None keeps the NFS file-drop
        # path (used by the in-process probes).
        self.endpoint = endpoint
        self._sock: Any = None
        self._next_id = 0
        if endpoint is None:
            self.ipc_dir.mkdir(parents=True, exist_ok=True)
            self._next_id = self._max_existing_id() + 1

    def _max_existing_id(self) -> int:
        ids = []
        for pattern in ("cmd_*.json", "resp_*.json"):
            for path in self.ipc_dir.glob(pattern):
                try:
                    ids.append(int(path.stem.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
        return max(ids, default=0)

    # ---------------------------------------------------------------- public
    def reset(self, init_state_idx: int = 0) -> Response:
        return self._call({"cmd": "reset", "init_state_idx": init_state_idx})

    def act(self, action: dict[str, Any]) -> Response:
        return self._call({"cmd": "act", "action": action})

    def look(self, camera: str = "agent") -> Response:
        return self._call({"cmd": "look", "camera": camera})

    def state(self) -> Response:
        return self._call({"cmd": "state"})

    def success(self) -> bool:
        resp = self._call({"cmd": "success"})
        return bool(resp.result.get("success", False))

    def ping(self) -> Response:
        return self._call({"cmd": "ping"})

    def close(self) -> None:
        try:
            self._call({"cmd": "close"}, timeout_s=30.0)
        except (SimClientError, OSError):
            pass
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # -------------------------------------------------------------------- tcp
    def _connect(self) -> None:
        """Open the TCP link to the sim server.

        WHY TCP NOW (this supersedes the old "TCP unreachable" note in
        server.py): that note was measured against 172.18.34.26 -- a DIFFERENT
        subnet with no route from gpu026. login01 also carries
        192.168.82.239/23, the SAME /23 as gpu026 (192.168.82.46), and that
        path answers in 0.67 ms. The NFS file-drop path costs 30 s per call:
        NFS `acdirmin` defaults to 30 s, so a command the client writes stays
        invisible to the server's directory glob for up to 30 s (measured:
        server-side work 0.56 s vs client-side wait 30.0 s).
        """
        assert self.endpoint is not None
        host, port = self.endpoint
        self._sock = socket.create_connection((host, port), timeout=self.timeout_s)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _call_tcp(self, command: dict[str, Any], timeout: float) -> Response:
        if self._sock is None:
            self._connect()
        assert self._sock is not None
        self._sock.settimeout(timeout)
        payload = json.dumps(command).encode("utf-8")
        self._sock.sendall(struct.pack(">I", len(payload)) + payload)
        size = struct.unpack(">I", _recv_exact(self._sock, 4))[0]
        resp = json.loads(_recv_exact(self._sock, size))
        response = Response(
            id=resp["id"],
            ok=bool(resp["ok"]),
            result=resp.get("result", {}),
            error=resp.get("error", ""),
        )
        if not response.ok:
            raise SimClientError(
                f"command {resp['id']} ({command.get('cmd')}) failed: {response.error}"
            )
        return response

    # ---------------------------------------------------------------- private
    def _call(self, command: dict[str, Any], timeout_s: float | None = None) -> Response:
        timeout = timeout_s if timeout_s is not None else self.timeout_s
        if self.endpoint is not None:
            return self._call_tcp(command, timeout)
        cmd_id = self._next_id
        self._next_id += 1

        cmd_path = self.ipc_dir / f"cmd_{cmd_id}.json"
        tmp_path = cmd_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(command))
        os.rename(tmp_path, cmd_path)  # atomic publish

        resp_path = self.ipc_dir / f"resp_{cmd_id}.json"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if resp_path.exists():
                # The server publishes via rename, so the file is complete.
                payload = json.loads(resp_path.read_text())
                resp_path.unlink(missing_ok=True)
                response = Response(
                    id=payload["id"],
                    ok=bool(payload["ok"]),
                    result=payload.get("result", {}),
                    error=payload.get("error", ""),
                )
                if not response.ok:
                    raise SimClientError(
                        f"command {cmd_id} ({command.get('cmd')}) failed: {response.error}"
                    )
                return response
            time.sleep(0.01)
        raise SimClientError(
            f"command {cmd_id} ({command.get('cmd')}) timed out after {timeout:.0f}s"
        )
