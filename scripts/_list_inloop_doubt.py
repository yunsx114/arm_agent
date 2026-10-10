"""Did the model ever doubt its alignment inside the loop?

The single-turn probe shows the model CAN judge "篮子不在正下方" from the same
frames.  This checks whether any of that doubt ever surfaced in the 80-turn run,
which decides between "it judged and then acted anyway" and "the judging step
never ran in the tool-calling regime".
"""

from __future__ import annotations

import json
import sys

EPISODE = sys.argv[1] if len(sys.argv) > 1 else "outputs/episodes/episode_000_1791623248.json"
DOUBT = ("不在", "没对准", "未对准", "偏", "还没到", "不够", "距离", "再移", "靠近一点", "继续移")

d = json.load(open(EPISODE))
hits = 0
for e in d["transcript"]:
    if "assistant" not in e:
        continue
    body = e["assistant"].split("<tool_call>")[0].strip()
    if any(k in body for k in DOUBT):
        hits += 1
        print(f"t{e['turn']:>3}  {body[:150]!r}")
print(f"\n含方位存疑措辞的轮数: {hits} / 80")

claims = sum(
    1
    for e in d["transcript"]
    if "assistant" in e and "已到达篮子正上方" in e["assistant"]
)
print(f'断言"已到达篮子正上方"的轮数: {claims}')
