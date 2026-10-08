"""Phase 0 · E4 probe: can a GPU node open a TCP connection to the login node?

This decides Plan B's transport: if TCP works we run a long-lived sim server on
the login node; if not we fall back to NFS file-drop polling.

Usage (two terminals / one shell + one srun):

  # on the login node (server), stays up until one client is served:
  python3 scripts/probe_tcp_e4.py server --port 9876 --timeout 180

  # from a GPU node (client):
  srun --partition=rtx2080ti --account=gpulab02 --qos=rtx2080ti --nodes=1 \
       --gres=gpu:1 --time=5 --job-name=e4_tcp \
       /lab/haoq_lab/cse12311731/miniconda3/envs/libero/bin/python \
       scripts/probe_tcp_e4.py client --host 172.18.34.26 --port 9876

Server prints the peer address and echoes a line back; client prints the reply.
Exit 0 = connectivity works.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time


def run_server(host: str, port: int, timeout: float) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        srv.settimeout(timeout)
        print(f"[e4] server listening on {host}:{port} (timeout {timeout}s)", flush=True)
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            print("[e4] server timed out waiting for a client")
            return 1
        with conn:
            print(f"[e4] accepted connection from {addr[0]}:{addr[1]}", flush=True)
            data = conn.recv(4096)
            print(f"[e4] received: {data!r}")
            reply = b"ack from login node at " + str(time.time()).encode()
            conn.sendall(reply)
            print("[e4] replied, closing")
    return 0


def run_client(host: str, port: int) -> int:
    payload = f"hello from {socket.gethostname()} at {time.time()}"
    start = time.time()
    with socket.create_connection((host, port), timeout=20) as sock:
        sock.sendall(payload.encode())
        sock.settimeout(20)
        reply = sock.recv(4096)
    dt = (time.time() - start) * 1000
    print(f"[e4] sent: {payload}")
    print(f"[e4] received: {reply!r}")
    print(f"[e4] round trip: {dt:.0f} ms")
    print("[e4] TCP OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["server", "client"])
    parser.add_argument("--host", default="172.18.34.26")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.mode == "server":
        return run_server("0.0.0.0", args.port, args.timeout)
    return run_client(args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
