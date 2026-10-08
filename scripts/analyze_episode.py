"""Print an episode transcript as a compact timeline for review.

    bash scripts/simenv.sh scripts/analyze_episode.py [path/to/episode.json]
    (or any python; this script needs no sim dependencies)

Defaults to the newest transcript in outputs/episodes/. Output: one line per
turn (tool, key args, the measured numbers the model saw), plus the result
summary.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EPISODES = REPO / "outputs" / "episodes"

_MOVE_RE = re.compile(r"实测位移 \(([-+0-9.]+), ([-+0-9.]+), ([-+0-9.]+)\)")
_STATE_RE = re.compile(r"机械爪(?:位置|现在位于) \(([-+0-9.]+), ([-+0-9.]+), ([-+0-9.]+)\)")


def main() -> int:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        candidates = sorted(EPISODES.glob("episode_*.json"))
        if not candidates:
            print(f"no transcripts in {EPISODES}")
            return 1
        path = candidates[-1]

    payload = json.loads(path.read_text())
    result = payload["result"]
    print(f"===== {path.name} =====")
    print(
        f"task={result['task']}\nlang={result['language']}\n"
        f"success={result['success']} declared={result['done_declared']} "
        f"turns={result['turns']} calls={result['tool_calls']} "
        f"retries={result['protocol_retries']} elapsed={result['elapsed_s']}s"
    )
    print("\n--- timeline ---")
    for entry in payload["transcript"]:
        turn = entry.get("turn", "?")
        if "tool_error" in entry:
            print(f"[{turn:>3}] ERROR: {entry['tool_error'][:100]}")
            continue
        if entry.get("protocol_retry"):
            print(f"[{turn:>3}] protocol retry")
            continue
        assistant = entry.get("assistant", "")
        # first <function=...> and its parameters
        fn = re.search(r"<function=([\w.\-]+)>", assistant)
        params = re.findall(r"<parameter=([\w.\-]+)>\s*(.*?)\s*</parameter>", assistant, re.DOTALL)
        call = f"{fn.group(1)}({', '.join(f'{k}={v}' for k, v in params)})" if fn else "(no call)"
        outcome = entry.get("result", "")
        delta = _MOVE_RE.search(outcome)
        if delta:
            dx, dy, dz = (float(delta.group(i)) for i in (1, 2, 3))
            outcome = f"Δ=({dx*100:+.1f},{dy*100:+.1f},{dz*100:+.1f})cm | " + outcome.split("；")[-1]
        print(f"[{turn:>3}] {call:<44s} -> {outcome[:110]}")

    print("\n--- last assistant text ---")
    for entry in reversed(payload["transcript"]):
        if entry.get("assistant"):
            print(entry["assistant"][:400])
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
