"""Did the `aim` ring actually make the model's pointing converge?

WHY THIS SCRIPT EXISTS (档 B v2). The design bet is that a model which cannot NAME an
absolute pixel (measured: 30-50 px systematic bias at both 256 px and 512 px, stable
across calls) can still JUDGE the gap between a drawn ring and an object, and close it
in one or two rounds. That bet is cheap to falsify and expensive to guess at, so this
turns the server's `[aim]` instrument into the number that decides it:

  * Δ_nearest  -- world xy distance from the grounded point to the NEAREST object
  * Δ_can      -- world xy distance from the grounded point to `alphabet_soup_1`
                  (task 0's target; the acceptance box is ±6.1 cm around the basket
                  opening and the grasp tolerance measured ±2.2 cm)

Read as: if Δ_can falls monotonically toward the ~1 cm the grounding is capable of,
the ring is usable and the model is using it. If it hovers at the model's 30 px bias,
`aim` bought nothing and the next step is a coarser interface (a candidate menu or a
per-axis `negative|hold|positive`), not a finer image.

Episodes are split on the server log's `reset` commands, so the agent log and the server
log do not have to be timestamped against each other.

Usage:
    python3 scripts/audit_aim_convergence.py outputs/logs/single_node_server.log
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# `[aim] tool=aim asked=(96,132) cam=agent target=[ 0.066 -0.102  0.118] depth=0.986m
#  :: salad_dressing_1 px=(98, 141) xy=17mm dz=+45mm | milk_1 px=(66, 151) xy=101mm ...`
AIM_LINE = re.compile(
    r"^\[aim\] tool=(?P<tool>\w+) asked=\((?P<u>-?\d+), ?(?P<v>-?\d+)\) cam=(?P<cam>\w+)"
)
REJECT_LINE = re.compile(r"^\[aim\] tool=(?P<tool>\w+) asked=\(.*?\) cam=(?P<cam>\w+) REJECTED")
ENTRY = re.compile(r"(?P<name>\w+?) px=\((?P<pu>\d+), ?(?P<pv>\d+)\) xy=(?P<xy>\d+)mm")


def parse(path: Path) -> list[dict]:
    episodes: list[dict] = []
    current: dict | None = None
    for raw in path.read_text(errors="replace").splitlines():
        if raw.startswith("[server] cmd") and " reset " in raw:
            current = {"aims": []}
            episodes.append(current)
            continue
        if current is None or not raw.startswith("[aim]"):
            continue
        rejected = REJECT_LINE.match(raw)
        head = AIM_LINE.match(raw)
        if head is None and rejected is None:
            continue
        tool = (head or rejected).group("tool")
        cam = (head or rejected).group("cam")
        entries = {
            m.group("name"): (int(m.group("xy")), (int(m.group("pu")), int(m.group("pv"))))
            for m in ENTRY.finditer(raw)
        }
        current["aims"].append(
            {
                "tool": tool,
                "cam": cam,
                "asked": (
                    (int(head.group("u")), int(head.group("v"))) if head else None
                ),
                "rejected": bool(rejected),
                "nearest": min(entries.items(), key=lambda kv: kv[1][0]) if entries else None,
                "can": entries.get("alphabet_soup_1"),
            }
        )
    return episodes


def main(argv: list[str]) -> int:
    path = Path(argv[1] if len(argv) > 1 else "outputs/logs/single_node_server.log")
    if not path.exists():
        print(f"no such log: {path}")
        return 2
    episodes = parse(path)
    print(f"{path}: {len(episodes)} episode(s) with pixel calls\n")
    for i, ep in enumerate(episodes):
        aims = ep["aims"]
        if not aims:
            continue
        can_deltas = [a["can"][0] for a in aims if a.get("can")]
        near_deltas = [a["nearest"][1][0] for a in aims if a.get("nearest")]
        targets = [a for a in aims if a["tool"] == "target"]
        print(f"== episode {i}: {len(aims)} pixel calls ({len(targets)} target)")
        print("   agent-view Δ_can (mm), in order: " + ", ".join(str(d) for d in can_deltas))
        print("   Δ_nearest (mm), in order:        " + ", ".join(str(d) for d in near_deltas))
        if can_deltas:
            print(f"   first {can_deltas[0]} mm -> last {can_deltas[-1]} mm "
                  f"(min {min(can_deltas)})")
        for a in targets:
            can = a.get("can")
            print(
                f"   target cam={a['cam']} asked={a['asked']}"
                + (f" Δ_can={can[0]}mm can_px={can[1]}" if can else " (no can entry)")
            )
        print()
    all_can = [a["can"][0] for ep in episodes for a in ep["aims"] if a.get("can")]
    if all_can:
        print(f"ALL agent-view pixel calls: n={len(all_can)} min={min(all_can)}mm "
              f"median={sorted(all_can)[len(all_can) // 2]}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
