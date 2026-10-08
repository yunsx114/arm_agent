"""Two transports for the sim server (runs in the `libero` env, login node).

**TCP is the production transport** (`--tcp-port`). It was originally ruled out
by a measurement against 172.18.34.26 -- a different subnet gpu026 cannot
route to. login01 also carries **192.168.82.239/23**, the SAME /23 as gpu026
(192.168.82.46), and that path answers in **0.67 ms** (re-measured 2026-10-08).

The NFS file-drop path is kept for the in-process probes only. It costs **30 s
per call** in the real two-node setup: NFS `acdirmin` defaults to 30 s, so a
command the client publishes stays invisible to the server's directory glob for
up to 30 s. Measured breakdown of one `move`: server-side work 0.56 s,
client-side wait 30.0 s.

Protocol (TCP; 4-byte big-endian length prefix + UTF-8 JSON, frames as base64):
    cmd -> server    {"cmd": "act", "action": {...}}
    server -> cmd    {"id": N, "ok": true, "result": {..., "images_b64": {...}}}

NFS protocol (all paths under <root>/runtime/ipc), for reference:
    client -> server   cmd_<id>.json        published via atomic rename
    server -> client   resp_<id>.json       published via atomic rename
                       img_<id>_<cam>.png   camera frames, if the command made any
"""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import struct
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from arm_agent import contracts as C
from arm_agent.sim.action_adapter import ActionAdapter
from arm_agent.sim.libero_env import LiberoSim


class SimServer:
    def __init__(
        self,
        ipc_dir: Path,
        suite_name: str = "libero_object",
        task_id: int = 0,
        cam_size: int = 256,
        seed: int = 0,
    ) -> None:
        self.ipc_dir = Path(ipc_dir)
        self.ipc_dir.mkdir(parents=True, exist_ok=True)
        self.sim = LiberoSim(suite_name=suite_name, task_id=task_id, cam_size=cam_size, seed=seed)
        self.adapter = ActionAdapter(self.sim)
        self._images_saved = 0
        self._closing = False

    # ------------------------------------------------------------------ main
    def serve_forever(self, poll_s: float = C.IPC_POLL_INTERVAL_S, idle_exit_s: float = 3600.0) -> None:
        print(f"[server] listening on {self.ipc_dir} (poll {poll_s * 1000:.0f} ms)", flush=True)
        last_activity = time.time()
        while True:
            handled = self._handle_pending()
            if handled:
                last_activity = time.time()
            else:
                if time.time() - last_activity > idle_exit_s:
                    print("[server] idle timeout, exiting", flush=True)
                    break
                time.sleep(poll_s)
            if self._closing:
                break
        self.sim.close()
        print("[server] stopped", flush=True)

    # -------------------------------------------------------------- commands
    def _handle_pending(self) -> bool:
        handled = False
        for cmd_path in sorted(self.ipc_dir.glob(C.IPC_CMD_GLOB)):
            try:
                cmd_id = int(cmd_path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                cmd_path.unlink(missing_ok=True)
                continue
            try:
                cmd = json.loads(cmd_path.read_text())
                t0 = time.time()
                result = self._dispatch(cmd, cmd_id)
                dt = time.time() - t0
                resp: dict[str, Any] = {"id": cmd_id, "ok": True, "result": result}
                print(f"[server] cmd {cmd_id} {cmd.get('cmd')} ok in {dt:.2f}s", flush=True)
            except Exception as exc:  # noqa: BLE001 - a bad command must not kill the server
                resp = {
                    "id": cmd_id,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(f"[server] cmd {cmd_id} FAILED: {exc}", flush=True)
            self._write_atomic(self.ipc_dir / f"resp_{cmd_id}.json", json.dumps(resp))
            cmd_path.unlink(missing_ok=True)
            handled = True
        return handled

    def _dispatch(
        self, cmd: dict[str, Any], cmd_id: int, inline_images: bool = False
    ) -> dict[str, Any]:
        kind = cmd.get("cmd")
        if kind == "ping":
            return {"pong": True, "task": self.sim.task.name}
        if kind == "reset":
            obs = self.sim.reset(init_state_idx=int(cmd.get("init_state_idx", 0)))
            return {"obs": obs, **self._image_payload(cmd_id, inline_images)}
        if kind == "act":
            action = cmd.get("action", {})
            act_kind = action.get("kind")
            if act_kind == "move":
                outcome = self.adapter.move(
                    str(action["direction"]), float(action["distance_cm"])
                )
            elif act_kind == "rotate":
                outcome = self.adapter.rotate(str(action["axis"]), float(action["angle_deg"]))
            elif act_kind == "gripper":
                outcome = self.adapter.set_gripper(str(action["action"]))
            elif act_kind == "align":
                # Server-side primitive: servo the EEF x/y onto an object.
                # See ActionAdapter.align_xy for why the LLM cannot do this.
                name = str(action["name"])
                # NOTE (fixed): "objects" is SYNTHESISED by pack() from the raw
                # `<name>_pos` keys; it does NOT exist in sim._obs. Reading it
                # off _obs returned {} every time, so align/descend always
                # raised "no object matching ...; have []" (measured: a whole
                # 20-turn episode wasted on that error).
                objects = self.sim.pack(self.sim._obs)["objects"] if self.sim._obs else {}
                key = self._match_object(name, objects)
                if key is None:
                    raise ValueError(f"no object matching {name!r}; have {sorted(objects)}")
                pos = objects[key]
                z_safe = float(action.get("z_safe", 0.22))
                outcome = self.adapter.align_xy((pos[0], pos[1]), z_safe=z_safe)
            elif act_kind == "descend":
                # Aligned descent onto an object: all three axes corrected each
                # tick (align_xy + separate -z was measured to block at can-top
                # height and close on air).
                name = str(action["name"])
                # Same fix as the align branch: take objects from pack(), not
                # from the raw _obs (which never has an "objects" key).
                objects = self.sim.pack(self.sim._obs)["objects"] if self.sim._obs else {}
                key = self._match_object(name, objects)
                if key is None:
                    raise ValueError(f"no object matching {name!r}; have {sorted(objects)}")
                pos = objects[key]
                z_offset_cm = float(action.get("z_offset_cm", 1.2))
                z_target = pos[2] + z_offset_cm / 100.0
                outcome = self.adapter.descend_to((pos[0], pos[1]), z_target)
            else:
                raise ValueError(f"unknown action kind: {act_kind!r}")
            obs = self.sim.pack(self.sim._obs)  # type: ignore[arg-type]
            return {
                "outcome": outcome.to_dict(),
                "obs": obs,
                **self._image_payload(cmd_id, inline_images),
            }
        if kind == "look":
            return {
                "obs": self.sim.pack(self.sim._obs),  # type: ignore[arg-type]
                **self._image_payload(cmd_id, inline_images),
            }
        if kind == "state":
            return {"obs": self.sim.pack(self.sim._obs)}  # type: ignore[arg-type]
        if kind == "success":
            return {"success": self.sim.check_success()}
        if kind == "close":
            self._closing = True
            return {"closing": True}
        raise ValueError(f"unknown command: {kind!r}")

    # ------------------------------------------------------------------ util
    @staticmethod
    def _match_object(query: str, objects: dict[str, Any]) -> str | None:
        """Fuzzy object lookup, shared by locate (agent side) and align (here)."""
        query_norm = str(query).strip().lower().replace(" ", "_")
        if query_norm in objects:
            return query_norm
        for key in objects:
            stem = key.rsplit("_", 1)[0].lower()
            if query_norm and (query_norm in key.lower() or query_norm in stem or stem in query_norm):
                return key
        return None

    def _image_payload(self, cmd_id: int, inline: bool) -> dict[str, Any]:
        """Camera frames for a response: base64-inline (TCP) or on disk (NFS)."""
        if inline:
            return {"images_b64": self._images_b64()}
        return {"images": self._save_images(cmd_id)}

    def _images_b64(self) -> dict[str, str]:
        """Both camera frames as base64 PNGs (display orientation)."""
        from PIL import Image

        self.sim.refresh_images()
        out: dict[str, str] = {}
        for cam, frame in self.sim.images().items():
            buf = io.BytesIO()
            Image.fromarray(np.asarray(frame)[::-1]).save(buf, format="PNG")
            out[cam] = base64.b64encode(buf.getvalue()).decode("ascii")
        return out

    def _save_images(self, cmd_id: int) -> dict[str, str]:
        """Persist both camera frames as PNG; returns camera -> relative path."""
        from PIL import Image

        # Idempotent: renders only if the cameras were disabled during internal
        # ticks (or never refreshed); otherwise it is a no-op.
        self.sim.refresh_images()
        paths: dict[str, str] = {}
        for cam, frame in self.sim.images().items():
            name = f"img_{cmd_id}_{cam}.png"
            # Frames are stored vertically flipped for display (LIBERO
            # convention); flip once here so every consumer sees the same
            # display orientation the E6 probe validated.
            Image.fromarray(np.asarray(frame)[::-1]).save(self.ipc_dir / name)
            paths[cam] = name
        self._images_saved += 1
        return paths

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.rename(tmp, path)  # NFS rename is atomic (measured)

    # ------------------------------------------------------------------ tcp
    def serve_tcp(self, port: int, idle_exit_s: float = 3600.0) -> None:
        """Serve the protocol over TCP (production path).

        Accepts clients one at a time and processes commands strictly in order,
        mirroring the file-drop semantics. Frames travel base64-inline, so
        neither side ever waits on NFS attribute-cache expiry (that cost 30 s
        per call, measured). Accepting in a LOOP lets a short-lived liveness
        probe (`cli.py ipc-ping`) connect and disconnect without ending the
        session.
        """
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(4)
        print(f"[server] TCP listening on 0.0.0.0:{port}", flush=True)
        cmd_id = 0
        try:
            while not self._closing:
                srv.settimeout(max(1.0, idle_exit_s))
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    print("[server] no client arrived, idle exit", flush=True)
                    break
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"[server] client connected from {addr}", flush=True)
                try:
                    while not self._closing:
                        conn.settimeout(max(1.0, idle_exit_s))
                        header = conn.recv(4)
                        if not header:
                            print("[server] client disconnected", flush=True)
                            break
                        while len(header) < 4:
                            more = conn.recv(4 - len(header))
                            if not more:
                                raise ConnectionResetError("header truncated")
                            header += more
                        size = struct.unpack(">I", header)[0]
                        body = b""
                        while len(body) < size:
                            chunk = conn.recv(size - len(body))
                            if not chunk:
                                raise ConnectionResetError("body truncated")
                            body += chunk
                        cmd = json.loads(body)
                        cmd_id += 1
                        t0 = time.time()
                        try:
                            result = self._dispatch(cmd, cmd_id, inline_images=True)
                            resp: dict[str, Any] = {"id": cmd_id, "ok": True, "result": result}
                            print(
                                f"[server] cmd {cmd_id} {cmd.get('cmd')} ok in "
                                f"{time.time() - t0:.3f}s",
                                flush=True,
                            )
                        except Exception as exc:  # noqa: BLE001 - a bad command must not kill the server
                            resp = {
                                "id": cmd_id,
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                            }
                            print(f"[server] cmd {cmd_id} FAILED: {exc}", flush=True)
                        out = json.dumps(resp).encode("utf-8")
                        conn.sendall(struct.pack(">I", len(out)) + out)
                except (ConnectionResetError, socket.timeout, OSError) as exc:
                    print(f"[server] connection ended: {type(exc).__name__}", flush=True)
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
        finally:
            srv.close()
            self.sim.close()
            print("[server] stopped", flush=True)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="arm_agent sim server (login node)")
    parser.add_argument("--ipc-dir", required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--cam", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--idle-exit", type=float, default=3600.0)
    parser.add_argument(
        "--tcp-port",
        type=int,
        default=None,
        help="serve over TCP on this port (production path); omit to use the NFS file-drop",
    )
    args = parser.parse_args(argv)

    server = SimServer(
        ipc_dir=Path(args.ipc_dir),
        suite_name=args.suite,
        task_id=args.task_id,
        cam_size=args.cam,
        seed=args.seed,
    )
    if args.tcp_port:
        server.serve_tcp(port=args.tcp_port, idle_exit_s=args.idle_exit)
    else:
        server.serve_forever(idle_exit_s=args.idle_exit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
