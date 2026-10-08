"""Encode a leftover frame directory into an mp4 (crash / timeout recovery).

The episode recorder writes one PNG per turn while the episode runs and only
encodes at finalize(); if the process died hard (SIGKILL, srun timeout, OOM
kill) the frames are still on disk. This script turns them into the same video
the recorder would have produced:

    bash scripts/agentenv.sh scripts/encode_frames.py \
        outputs/videos/ep000_1770000000/frames [--fps 2] [--out video.mp4]

Also handy to re-encode with a different fps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_dir", help="directory containing frame_*.png")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--out", default="", help="output mp4 (default: <parent>/video.mp4)")
    parser.add_argument("--quality", type=int, default=8, help="libx264 quality (higher=better)")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        print(f"no frame_*.png under {frames_dir}")
        return 1
    out = Path(args.out) if args.out else frames_dir.parent / "video.mp4"

    import imageio.v2 as imageio

    print(f"encoding {len(frames)} frames -> {out} @ {args.fps} fps")
    writer = imageio.get_writer(
        str(out),
        fps=args.fps,
        codec="libx264",
        quality=args.quality,
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        for path in frames:
            writer.append_data(imageio.imread(path))
    finally:
        writer.close()
    size_mb = out.stat().st_size / 1e6
    print(f"done: {out} ({size_mb:.1f} MB, {len(frames) / args.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
