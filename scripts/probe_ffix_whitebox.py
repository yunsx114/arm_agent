#!/usr/bin/env python3
"""White-box checks for the two 档 F fixes (30 s, stub model, real sim).

1. descent clamp: a `move -z 10` in look mode must execute 3 cm (contracts.DESCENT_CAP_CM),
   carry descent_clamped_from_cm=10 in the structured measurement, and append the 档 F
   note to the text the model reads.
2. retry refund: two tool-less replies (finish=length style) must NOT count toward
   turns: with max_turns=4 and 2 productive calls + 2 retries, result.turns must be 2.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path

REPO = Path("/root/arm_agent")
sys.path.insert(0, str(REPO / "src"))

from arm_agent import contracts as C  # noqa: E402
from arm_agent.agent.harness import Harness  # noqa: E402
from arm_agent.agent.strata import to_xml_tool_call  # noqa: E402
from arm_agent.sim.client import SimClient  # noqa: E402
from arm_agent.sim.server import SimServer  # noqa: E402

import json as _json


def _call(name: str, args: dict) -> str:
    """The exact wire form `to_xml_tool_call` emits (what the parser expects)."""
    return to_xml_tool_call(
        {"function": {"name": name, "arguments": _json.dumps(args)}}
    )


MOVE_CALL = "\n现状：在罐上方。判定：下降。\n" + _call("move", {"direction": "-z", "distance_cm": "10"})
GRIP_CALL = "\n判定：张开。\n" + _call("gripper", {"action": "open"})


class Stub:
    def __init__(self):
        self.script = [MOVE_CALL, "我还在思考，尚未决定动作。", "(thinking continues…)", GRIP_CALL]

    def chat(self, messages, tools=None, **kw):  # noqa: ANN001, ANN003
        return self.script.pop(0)


def main() -> int:
    ipc = REPO / "runtime" / "ipc_ffix"
    shutil.rmtree(ipc, ignore_errors=True)
    server = SimServer(ipc_dir=ipc, suite_name="libero_object", task_id=0,
                       cam_size=128, observation_mode="look")
    threading.Thread(target=server.serve_forever, kwargs={"idle_exit_s": 60}, daemon=True).start()
    time.sleep(0.3)
    client = SimClient(ipc)
    harness = Harness(client=client, model=Stub(), workspace=REPO,
                      max_turns=4, tool_protocol="v2", log=False,
                      observation_mode="look", summary_every=0)
    result = harness.run_episode(init_state_idx=0, episode_id=990)
    client.close()

    import json
    t = json.loads(Path(result.transcript_path).read_text())["transcript"]
    moves = [e for e in t if (e.get("structured_result") or {}).get("tool") == "move"]
    ok = True
    if not moves:
        print("FAIL: no move executed"); return 1
    m = moves[0]["structured_result"]["measurement"]
    delta_z = m.get("measured_delta_m", [0, 0, 0])[2]
    clamped = m.get("descent_clamped_from_cm")
    text = str(moves[0].get("result", ""))
    print(f"move: requested 10cm → delta_z={delta_z} (cap {C.DESCENT_CAP_CM}), clamped_from={clamped}")
    print(f"text contains 档 F note: {'档 F' in text}")
    ok &= abs(delta_z + C.DESCENT_CAP_CM / 100.0) < 2e-3 and clamped == 10.0 and "档 F" in text
    print(f"turns={result.turns} (expect 2), tool_calls={result.tool_calls} (expect 2), "
          f"retries={result.protocol_retries} (expect 2)")
    ok &= result.turns == 2 and result.tool_calls == 2 and result.protocol_retries == 2
    print("ALL OK" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
