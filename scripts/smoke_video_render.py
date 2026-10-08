"""Local smoke of the video renderer: rebuild a video from an existing transcript.

No GPU / sim involved — validates layout, CJK font, wrapping and encoding from
artifacts already on disk:

    bash scripts/agentenv.sh scripts/smoke_video_render.py [transcript.json]

Uses the E1 smoke images as stand-in camera frames. Prints the mp4 path and
extracts one frame to PNG for visual inspection.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from arm_agent.agent.render import EpisodeRecorder  # noqa: E402

_FN_RE = re.compile(r"<function=([\w.\-]+)>")
_PARAM_RE = re.compile(r"<parameter=([\w.\-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def main() -> int:
    if len(sys.argv) > 1:
        transcript_path = Path(sys.argv[1])
    else:
        candidates = sorted((REPO / "outputs" / "episodes").glob("episode_*.json"))
        if not candidates:
            print("no transcript found")
            return 1
        transcript_path = candidates[-1]
    payload = json.loads(transcript_path.read_text())
    print(f"transcript: {transcript_path.name} (title: {payload['result'].get('task')})")

    from PIL import Image

    agent_img = Image.open(REPO / "outputs" / "smoke" / "e1_libero_object_0_agentview.png")
    wrist_img = Image.open(REPO / "outputs" / "smoke" / "e1_libero_object_0_wrist.png")
    images = {"agent": agent_img, "wrist": wrist_img}

    recorder = EpisodeRecorder(
        video_dir=REPO / "outputs" / "videos",
        episode_id=777,
        task=payload["result"].get("language", ""),
        fps=4,
        keep_frames=True,  # keep for inspection in this smoke
    )
    recorder._last_images = images

    # Transcripts store the assistant reply and the tool result as SEPARATE
    # entries per turn (the harness appends them at different points); the live
    # recorder gets both together, so merge them here for a representative view.
    merged: dict[int, dict] = {}
    for entry in payload["transcript"]:
        turn = entry.get("turn", 0)
        slot = merged.setdefault(turn, {})
        if entry.get("assistant"):
            slot["assistant"] = entry["assistant"]
        if entry.get("result"):
            slot["result"] = entry["result"]
        if entry.get("tool_error"):
            slot["result"] = entry["tool_error"]
        if entry.get("protocol_retry"):
            slot["result"] = slot.get("result", "(protocol retry)")

    for turn, slot in sorted(merged.items()):
        assistant = slot.get("assistant", "")
        fn = _FN_RE.search(assistant)
        call = ""
        if fn:
            params = _PARAM_RE.findall(assistant)
            call = f"{fn.group(1)}({', '.join(f'{k}={v}' for k, v in params)})"
        recorder.record(
            turn, reply=assistant, tool_call=call, result=slot.get("result", ""), images=images
        )

    res = payload["result"]
    video = recorder.finalize(
        success=bool(res.get("success")),
        done_declared=bool(res.get("done_declared")),
        turns=res.get("turns", 0),
        elapsed_s=res.get("elapsed_s", 0.0),
    )
    print(f"video: {video}")

    # Extract a middle frame for visual check.
    import imageio.v2 as imageio

    frames = sorted(recorder.frames_dir.glob("frame_*.png"))
    mid = frames[len(frames) // 2]
    img = imageio.imread(mid)
    out_png = recorder.run_dir / "check_frame.png"
    imageio.imwrite(out_png, img)
    print(f"check frame: {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
