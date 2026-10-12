"""arm_agent CLI. Two of the subcommands run in different conda envs:

  sim-server   `libero` env (login node): holds the LIBERO env, serves NFS IPC
  run          `qwen35` env (GPU node via srun): the LLM + harness loop
  ipc-ping     either env: is a sim server alive at this ipc dir?

See scripts/simenv.sh / scripts/agentenv.sh for the exact wrappers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # arm_agent/
DEFAULT_IPC = REPO / "runtime" / "ipc"
# The 19GB weights live NEXT TO the repo, not inside it, so derive them from the
# repo's parent (see the layout diagram in scripts/_common.sh). Overridable by
# env var so the repo is not tied to one cluster layout.
DEFAULT_MODEL = Path(
    os.environ.get(
        "ARM_AGENT_MODEL_DIR", REPO.parent / "qwen35_demo" / "models" / "Qwen3.5-9B"
    )
)


def cmd_sim_server(args: argparse.Namespace) -> int:
    from arm_agent.sim.server import SimServer

    server = SimServer(
        ipc_dir=Path(args.ipc_dir),
        suite_name=args.suite,
        task_id=args.task_id,
        cam_size=args.cam,
        seed=args.seed,
        observation_mode=args.observation_mode,
        pixel_grid_px=args.pixel_grid,
        aim_mark_px=args.aim_mark,
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

    if args.backend == "strata":
        # Strata serves Qwen3.8-Flash-Next over OpenAI-compatible HTTP
        # (127.0.0.1:8080). See arm_agent/agent/strata.py for the three
        # conversions this backend does. Kept as a branch rather than a
        # replacement so the 9B path stays runnable for A/B.
        from arm_agent.agent.strata import StrataAgentModel

        model = StrataAgentModel(
            base_url=args.strata_url,
            model=args.strata_model,
            max_new_tokens=args.max_new_tokens,
            reasoning_effort=args.reasoning_effort,
            thinking_effort=args.thinking_effort,
        )
    else:
        model = QwenAgentModel(
            model_dir=args.model_dir,
            load=args.load,
            gpu_mem_gib=args.gpu_mem_gib,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            n_gpu=args.gpus,
        )
    model.load_model()

    harness = Harness(
        client=client,
        model=model,
        workspace=REPO,
        max_turns=args.max_turns,
        attach_images=not args.no_images,
        prompt_mode=args.prompt_mode,
        tool_protocol=args.tool_protocol,
        history_messages=args.history_messages,
        image_keep=args.image_keep,
        observation_mode=args.observation_mode,
        move_ladder=args.move_ladder,
    )
    print(
        f"[run] backend={args.backend} "
        f"history_messages={harness.history_messages} "
        f"image_keep={harness.image_keep} prompt_mode={harness.prompt_mode} "
        f"tool_protocol={harness.tool_protocol} "
        f"observation_mode={harness.observation_mode} "
        f"move_ladder={harness.move_ladder} "
        f"thinking={model.enable_thinking} max_new_tokens={model.max_new_tokens}",
        flush=True,
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
            f"[episode {episode_id}] success={result.success} "
            f"source={result.success_source or '-'} turns={result.turns} "
            f"tool_calls={result.tool_calls} retries={result.protocol_retries} "
            f"elapsed={result.elapsed_s:.0f}s transcript={result.transcript_path}",
            flush=True,
        )
        if args.record:
            print(f"[episode {episode_id}] video: {recorder.video_path}", flush=True)

    wins = sum(r.success for r in results)
    # Split by where the verdict came from: "env" = the sim's own check_success
    # flipped (the official LIBERO criterion), "declared" = the agent's own
    # verify agreed. A run whose successes are ALL "env" still counts, but it
    # says the agent never noticed on its own -- worth seeing at a glance.
    by_src = {}
    for r in results:
        if r.success:
            by_src[r.success_source or "?"] = by_src.get(r.success_source or "?", 0) + 1
    print(
        f"\n[run] episodes: {len(results)}, success: {wins}/{len(results)}"
        + (f"  (env={by_src.get('env', 0)}, declared={by_src.get('declared', 0)})" if wins else ""),
        flush=True,
    )
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
        "--observation-mode",
        default="truth",
        choices=["truth", "pixels", "raw", "look"],
        help="truth = 允许按名字取仿真真值坐标；pixels = 断掉那条路，"
        "只接受 aim/target 给像素，并把带网格+夹爪准线的画面送给模型；"
        "raw = 连像素工具也断掉，只剩 move + gripper + verify（画面上只留夹爪准线）；"
        "look = 同 raw，但**不再每轮自动给图**，改由模型自己调 look 工具取图（且历史图不清理）",
    )
    p.add_argument(
        "--pixel-grid",
        type=int,
        default=32,
        help="pixels 模式下画面网格间距（像素）",
    )
    p.add_argument(
        "--aim-mark",
        type=int,
        default=6,
        help="pixels 模式下 aim 红圈半径（像素）：实验变量，"
        "6 是当前默认（3 太小、看不出红圈与目标的差距）",
    )
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
    p.add_argument(
        "--load",
        default="4bit",
        choices=["4bit", "8bit", "bf16", "bfloat16", "fp16", "fp32", "cpu", "auto"],
    )
    p.add_argument("--gpu-mem-gib", type=float, default=9.5)
    p.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="shard the model across N GPUs via device_map=auto (gpu026 offers >=2)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=384,
        help="per-turn reply budget. 160 was too small: the model's Chinese "
        "preamble ate it and the tool call was cut off mid-XML (18 of 19 "
        "retries in one 80-turn run).",
    )
    p.add_argument(
        "--backend",
        default="local",
        choices=["local", "strata"],
        help="local = in-process Qwen3.5-9B (transformers, needs the GPU here); "
        "strata = HTTP to a running Strata server (Qwen3.8-Flash-Next)",
    )
    p.add_argument("--strata-url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--strata-model", default="strata", help="the name Strata ignores")
    p.add_argument(
        "--reasoning-effort",
        default="none",
        choices=["none", "low", "medium", "high"],
        help="strata backend only: the executor's thinking level (the server's "
        "own default is high). none keeps the turn short like the 9B's thinking=off.",
    )
    p.add_argument(
        "--thinking-effort",
        default="high",
        choices=["none", "low", "medium", "high"],
        help="strata backend only: what `thinking=True` maps to (reasoning_effort)",
    )
    p.add_argument("--timeout", type=float, default=600.0, help="per-IPC-call timeout")
    p.add_argument("--wait-server", type=float, default=120.0)
    p.add_argument("--no-images", action="store_true", help="text-only ablation")
    p.add_argument(
        "--prompt-mode",
        default="full",
        choices=["full", "plain"],
        help="full = 累积了场景专用补丁的提示；plain = 只给任务/坐标系/工具语义（P0a 剩离实验）",
    )
    p.add_argument(
        "--tool-protocol",
        default="v2",
        choices=["v1", "v2"],
        help="model tool surface; v1 keeps the legacy tools for A/B comparison",
    )
    p.add_argument(
        "--history-messages",
        type=int,
        default=None,
        help="conversation cap; omit for harness.MAX_HISTORY_MESSAGES (32 GiB tier). "
        "The 10 GiB tier is `--history-messages 4`.",
    )
    p.add_argument(
        "--image-keep",
        type=int,
        default=None,
        help="how many recent frames stay as pixels; -1 = 一张也不清理（look 模式的默认值）；"
        "omit for harness.HISTORY_IMAGE_KEEP",
    )
    p.add_argument("--record", action="store_true", help="record an mp4 per episode (cameras + model I/O)")
    p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="turn on the chat template's reasoning block. Off by default: it costs "
        "1-3k tokens/turn (~30-70 s here) and can eat max-new-tokens before the "
        "tool call is emitted. Raise --max-new-tokens when using it.",
    )
    p.add_argument(
        "--observation-mode",
        default="truth",
        choices=["truth", "pixels", "raw", "look"],
        help="truth = 仿真提供物体坐标（A/B 基线）；pixels = 不提供任何物体坐标 ，"
        "模型用带网格的画面给 aim/target 传像素；raw = 只有 move/gripper/verify ；"
        "look = 同 raw，但画面要模型自己 look（不自动给、历史图不清理）。"
        "必须同时给 sim-server 同样的值。",
    )
    p.add_argument(
        "--pixel-grid",
        type=int,
        default=32,
        help="pixels 模式下画在画面上的网格间距（像素）。实测代价：4px≈2.8mm、"
        "8px≈9.5mm、16px≈32mm（第三人称）；32px 仅用于标坐标。",
    )
    p.add_argument("--video-fps", type=int, default=2)
    p.add_argument("--video-dir", default=str(REPO / "outputs" / "videos"))
    p.add_argument("--keep-frames", action="store_true", help="keep per-turn PNG frames after encoding")
    p.add_argument(
        "--move-ladder",
        action="store_true",
        help="档 D：把 move 的距离换成三个固定档位（far=10cm / near=1cm / fine=1mm）。"
        "模型只挑方向+档位，距离由代码决定；sim-server 不需要知道这个开关。",
    )
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
