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

    TWO metrics, because they answer different questions and I conflated them once:
      * `Δ_nearest` -- distance to the CLOSEST object of any kind: how precisely the
        model can put the ring on SOMETHING. This is the pointing-precision number.
      * `Δ_can` -- distance to the task's object: precision AND identification, i.e.
        it also fails when the model aims carefully at the wrong container. Reading
        Δ_can alone made a 4-px-precise wrong-object aim look like a pointing failure.
    `within_tol` counts calls inside the measured GRASP TOLERANCE (22 mm: 2.2 cm still
    grasps, 4.6 cm closes on air).
    """
    deltas = [a["can"][0] for ep in episodes for a in ep["aims"] if a.get("can")]
    nearest = [a["nearest"][1][0] for ep in episodes for a in ep["aims"] if a.get("nearest")]
    if not deltas and not nearest:
        return f"{label}: no agent-view calls with object entries", []
    parts = [label]
    for tag, values in (("Δ_nearest", nearest), ("Δ_can", deltas)):
        if not values:
            parts.append(f"{tag}: none")
            continue
        ordered = sorted(values)
        within = [d for d in values if d <= 22]
        parts.append(
            f"{tag}: n={len(values)} min={ordered[0]}mm p25={ordered[len(ordered) // 4]}mm "
            f"median={ordered[len(ordered) // 2]}mm | ≤22mm: {len(within)}/{len(values)}"
        )
    return " | ".join(parts), deltas


def grasps_from_transcripts(pattern: str) -> dict[str, list[str]]:
    """Which object each `grasp` actually closed on, per episode file.

    THE SECOND METRIC, and the one that separates two different failures the aim
    numbers cannot tell apart: a model can be perfectly precise AND pick the wrong
    container (measured: 6/6 aims within 20 mm of `salad_dressing_1` while reporting
    it as the soup can). The contact table names the body it closed on
    (`measurement.object_body`), which is simulator truth and therefore lives ONLY in
    the transcript -- never in a result the model sees.
    """
    import glob
    import json

    out: dict[str, list[str]] = {}
    for path in sorted(glob.glob(pattern)):
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        held: list[str] = []
        for entry in data.get("transcript") or []:
            result = entry.get("structured_result") or {}
            if str(result.get("tool")) != "grasp":
                continue
            body = str((result.get("measurement") or {}).get("object_body") or "")
            held.append(body or "(nothing)")
        if held:
            out[Path(path).name] = held
    return out


TARGET_OBJECT = "alphabet_soup_1"


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--episodes=")]
    ep_glob = next(
        (a.split("=", 1)[1] for a in argv[1:] if a.startswith("--episodes=")),
        "outputs/episodes/episode_[0-9][0-9][0-9]_*.json",
    )
    paths = [Path(p) for p in (args or ["outputs/logs/single_node_server.log"])]
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
    # Identification, from the transcripts: a precise aim at the WRONG object is not a
    # pointing failure, and the two have to be counted separately.
    grasps = grasps_from_transcripts(ep_glob)
    if grasps:
        total = sum(len(v) for v in grasps.values())
        correct = sum(1 for v in grasps.values() for body in v if body.startswith(TARGET_OBJECT))
        print(f"\n== 抓取落在哪个物体上（转录 measurement.object_body，真值）")
        for name, bodies in grasps.items():
            print(f"  {name}: {bodies}")
        print(f"  正确物体({TARGET_OBJECT}): {correct}/{total} 次抓取")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
