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


def summarise(label: str, episodes: list[dict]) -> tuple[str, list[int]]:
    """One line per log: the numbers that decide whether `aim` is usable.

    `within_tol` counts calls whose Δ is inside the measured GRASP TOLERANCE (22 mm:
    2.2 cm still grasps, 4.6 cm closes on air). That is the only threshold that
    matters -- a ring 30 mm off looks like a success and behaves like a miss.
    """
    deltas = [a["can"][0] for ep in episodes for a in ep["aims"] if a.get("can")]
    if not deltas:
        return f"{label}: no agent-view calls with a can entry", []
    ordered = sorted(deltas)
    within = [d for d in deltas if d <= 22]
    line = (
        f"{label}: n={len(deltas)} min={ordered[0]}mm "
        f"p25={ordered[len(ordered) // 4]}mm median={ordered[len(ordered) // 2]}mm "
        f"| within grasp tol (22mm): {len(within)}/{len(deltas)}"
    )
    return line, deltas


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in (argv[1:] or ["outputs/logs/single_node_server.log"])]
    rows: list[tuple[str, list[int]]] = []
    for path in paths:
        if not path.exists():
            print(f"no such log: {path}")
            continue
        episodes = parse(path)
        with_calls = [ep for ep in episodes if ep["aims"]]
        print(f"=== {path}: {len(episodes)} episode(s), {len(with_calls)} with pixel calls")
        for i, ep in enumerate(episodes):
            aims = ep["aims"]
            if not aims:
                continue
            can_deltas = [a["can"][0] for a in aims if a.get("can")]
            nearest_names = [
                a["nearest"][0] for a in aims if a.get("nearest") and a["tool"] == "aim"
            ]
            targets = [a for a in aims if a["tool"] == "target"]
            print(f"  episode {i}: {len(aims)} pixel calls ({len(targets)} target)")
            if nearest_names:
                counts = {n: nearest_names.count(n) for n in dict.fromkeys(nearest_names)}
                print(f"    nearest object per aim: {counts}")
            if can_deltas:
                print("    Δ_can (mm): " + ", ".join(str(d) for d in can_deltas))
                print(f"    first {can_deltas[0]} -> last {can_deltas[-1]} "
                      f"(min {min(can_deltas)})")
        rows.append(summarise(str(path), episodes))
        print()
    print("== 汇总")
    for line, _ in rows:
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
