"""arm_agent CLI. Two of the subcommands run in different conda envs:

  sim-server   `libero` env (login node): holds the LIBERO env, serves NFS IPC
  run          `qwen35` env (GPU node via srun): the LLM + harness loop
  ipc-ping     either env: is a sim server alive at this ipc dir?

See scripts/simenv.sh / scripts/agentenv.sh for the exact wrappers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # arm_agent/
DEFAULT_IPC = REPO / "runtime" / "ipc"
DEFAULT_MODEL = Path("/lab/haoq_lab/cse12311731/qwen35_demo/models/Qwen3.5-9B")


def cmd_sim_server(args: argparse.Namespace) -> int:
    from arm_agent.sim.server import SimServer

    server = SimServer(
        ipc_dir=Path(args.ipc_dir),
        suite_name=args.suite,
        task_id=args.task_id,
        cam_size=args.cam,
        seed=args.seed,
    )
    if getattr(args, "tcp_port", None):
        server.serve_tcp(port=args.tcp_port, idle_exit_s=args.idle_exit)
    else:
        server.serve_forever(idle_exit_s=args.idle_exit)
    return 0


def cmd_ipc_ping(args: argparse.Namespace) -> int:
    from arm_agent.sim.client import SimClient

    endpoint = (args.tcp_host, args.tcp_port) if getattr(args, "tcp_port", None) else None
    client = SimClient(Path(args.ipc_dir), timeout_s=5.0, endpoint=endpoint)
    resp = client.ping()
    print(json.dumps(resp.result, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from arm_agent.agent.harness import Harness
    from arm_agent.agent.llm import QwenAgentModel
    from arm_agent.agent.render import EpisodeRecorder
    from arm_agent.sim.client import SimClient, SimClientError

    ipc_dir = Path(args.ipc_dir)
    endpoint = (args.tcp_host, args.tcp_port) if args.tcp_port else None
    client = SimClient(ipc_dir, timeout_s=args.timeout, endpoint=endpoint)
    if endpoint:
        print(f"[run] TCP transport -> {endpoint[0]}:{endpoint[1]}", flush=True)

    # Wait for the sim server (it may still be building the env).
    print(f"[run] waiting for sim server at {ipc_dir} ...", flush=True)
    deadline = time.time() + args.wait_server
    while True:
        try:
            client.ping()
            print("[run] sim server is alive", flush=True)
            break
        except (SimClientError, OSError):
            if time.time() > deadline:
                print("[run] ERROR: no sim server (start it with `cli.py sim-server`)")
                return 2
            time.sleep(1.0)

    model = QwenAgentModel(
        model_dir=args.model_dir,
        load=args.load,
        gpu_mem_gib=args.gpu_mem_gib,
        max_new_tokens=args.max_new_tokens,
        n_gpu=args.gpus,
    )
    model.load_model()

    harness = Harness(
        client=client,
        model=model,
        workspace=REPO,
        max_turns=args.max_turns,
        attach_images=not args.no_images,
    )

    results = []
    for episode_id in range(args.episodes):
        init_idx = (args.init_state + episode_id) % 50
        print(f"\n===== episode {episode_id} (init_state {init_idx}) =====", flush=True)
        recorder = EpisodeRecorder(
            video_dir=args.video_dir,
            episode_id=episode_id,
            fps=args.video_fps,
            keep_frames=args.keep_frames,
            enabled=args.record,
        )
        result = harness.run_episode(init_state_idx=init_idx, episode_id=episode_id, recorder=recorder)
        results.append(result)
        print(
            f"[episode {episode_id}] success={result.success} turns={result.turns} "
            f"tool_calls={result.tool_calls} retries={result.protocol_retries} "
            f"elapsed={result.elapsed_s:.0f}s transcript={result.transcript_path}",
            flush=True,
        )
        if args.record:
            print(f"[episode {episode_id}] video: {recorder.video_path}", flush=True)

    wins = sum(r.success for r in results)
    print(f"\n[run] episodes: {len(results)}, success: {wins}/{len(results)}", flush=True)
    client.close()
    return 0 if wins else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arm_agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sim-server", help="run the LIBERO sim server (libero env, login node)")
    p.add_argument("--ipc-dir", default=str(DEFAULT_IPC))
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--cam", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--idle-exit", type=float, default=7200.0)
    p.add_argument(
        "--tcp-port",
        type=int,
        default=None,
        help="serve over TCP on this port (production); omit for the NFS file-drop",
    )
    p.set_defaults(func=cmd_sim_server)

    p = sub.add_parser("ipc-ping", help="check that a sim server is alive")
    p.add_argument("--ipc-dir", default=str(DEFAULT_IPC))
    p.add_argument("--tcp-host", default="127.0.0.1")
    p.add_argument("--tcp-port", type=int, default=None)
    p.set_defaults(func=cmd_ipc_ping)

    p = sub.add_parser("run", help="run the LLM agent against a sim server (qwen35 env, GPU)")
    p.add_argument("--ipc-dir", default=str(DEFAULT_IPC))
    p.add_argument(
        "--tcp-host",
        default="192.168.82.239",
        help="sim server host for the TCP transport (login node's /23-subnet IP)",
    )
    p.add_argument("--tcp-port", type=int, default=None, help="sim server TCP port; enables the fast path")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--init-state", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    p.add_argument("--load", default="4bit", choices=["4bit", "8bit", "fp16", "fp32", "cpu", "auto"])
    p.add_argument("--gpu-mem-gib", type=float, default=9.5)
    p.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="shard the model across N GPUs via device_map=auto (gpu026 offers >=2)",
    )
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--timeout", type=float, default=600.0, help="per-IPC-call timeout")
    p.add_argument("--wait-server", type=float, default=120.0)
    p.add_argument("--no-images", action="store_true", help="text-only ablation")
    p.add_argument("--record", action="store_true", help="record an mp4 per episode (cameras + model I/O)")
    p.add_argument("--video-fps", type=int, default=2)
    p.add_argument("--video-dir", default=str(REPO / "outputs" / "videos"))
    p.add_argument("--keep-frames", action="store_true", help="keep per-turn PNG frames after encoding")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
