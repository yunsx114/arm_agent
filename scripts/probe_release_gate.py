"""Smoke-test the verifier role on real decision points before spending a run.

Calls the ACTUAL harness path (`Harness._release_verdict`) with, for each turn N:
  * the frames the executor saw when it decided turn N  (video frame N-1)
  * the executor's real reply text at turn N
and prints the verdict next to the measured truth (how far outside the basket
opening the release would have been).

Usage:
    PYTHONNOUSERSITE=1 PYTHONPATH=src ARM_AGENT_MODEL_DIR=/dev/shm/arm_agent_model \
      /root/miniconda3/envs/qwen35/bin/python scripts/probe_release_gate.py \
      [--episode ...] [--video ...] [--turns 9,15,21,26,31,32] [--thinking]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from PIL import Image  # noqa: E402

AGENT_BOX = (160, 20, 480, 340)
WRIST_BOX = (160, 380, 480, 700)
SOURCE_SIDE = 256


def _prp():
    spec = importlib.util.spec_from_file_location(
        "probe_release_perception", REPO / "scripts" / "probe_release_perception.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="outputs/episodes/episode_000_1791626311.json")
    ap.add_argument("--video", default="outputs/videos/ep000_1791625655/video.mp4")
    ap.add_argument("--turns", default="9,15,21,26,31,32")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--load", default=os.environ.get("LOAD", "bf16"))
    ap.add_argument("--gpu-mem-gib", type=float, default=float(os.environ.get("GPU_MEM_GIB", 29)))
    args = ap.parse_args()

    episode = str(REPO / args.episode) if not Path(args.episode).is_absolute() else args.episode
    video = str(REPO / args.video) if not Path(args.video).is_absolute() else args.video
    turns = [int(t) for t in args.turns.split(",") if t.strip()]

    prp = _prp()
    prp.EPISODE = episode
    truth = prp.measured()
    d = json.load(open(episode))
    replies = {e["turn"]: e["assistant"] for e in d["transcript"] if "assistant" in e}

    import imageio.v3 as iio

    need = {t - 1 for t in turns}
    frames: dict[int, dict[str, Image.Image]] = {}
    for i, arr in enumerate(iio.imiter(video)):
        if i in need:
            img = Image.fromarray(arr)
            frames[i] = {
                "agent": img.crop(AGENT_BOX).resize((SOURCE_SIDE, SOURCE_SIDE), Image.LANCZOS),
                "wrist": img.crop(WRIST_BOX).resize((SOURCE_SIDE, SOURCE_SIDE), Image.LANCZOS),
            }
        if i > max(need):
            break

    from arm_agent.agent.harness import Harness
    from arm_agent.agent.llm import QwenAgentModel

    model = QwenAgentModel(
        model_dir=os.environ.get(
            "ARM_AGENT_MODEL_DIR", str(REPO.parent / "qwen35_demo" / "models" / "Qwen3.5-9B")
        ),
        load=args.load,
        gpu_mem_gib=args.gpu_mem_gib,
        max_new_tokens=512,
        temperature=0.0,
        image_max_side=160,
        enable_thinking=False,
    )
    model.load_model()

    # A real Harness object so the tested path is the shipped one; the sim client
    # is never touched by the verifier.
    harness = Harness(
        client=None, model=model, workspace=REPO, tool_protocol="v2", verifier=True,
        verifier_thinking=True if args.thinking else None,
    )
    harness._task_text = d["result"].get("language") or d["result"].get("task")

    print(f"episode: {args.episode}   thinking={bool(args.thinking)}")
    print(f"{'turn':>4} {'实测(该轮爪末端)':<34} {'裁决':<8} 理由")
    print("-" * 100)
    n_reject = 0
    for t in turns:
        if t not in replies or (t - 1) not in frames:
            print(f"{t:>4} 跳过（缺帧或回复）")
            continue
        approved, reason = harness._release_verdict(replies[t], frames[t - 1])
        n_reject += 0 if approved else 1
        print(f"{t:>4} {truth.get(t, '-'):<34} {'放行' if approved else '拒绝':<8} {reason[:90]}")
    print(f"\n拒绝 {n_reject}/{len(turns)}  (verifier 调用次数={harness.verifier_calls})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
