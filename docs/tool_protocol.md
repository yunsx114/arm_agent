# Tool protocol

The agent supports two tool surfaces:

- `v1`: the legacy surface, retained for regression comparisons.
- `v2`: the default CLI surface, with six tools and structured results.

## v2 tools

```text
observe   Read state or object evidence; never moves the arm.
approach  Move above a target and descend; never operates the gripper.
move      Execute an explicit relative translation.
rotate    Execute an explicit end-effector rotation.
gripper   Open or close the gripper and report measured width/state.
verify    Ask the simulator to verify task success.
```

Every successful dispatch adds a `result` envelope:

```json
{
  "protocol": "arm-agent-tools-v2",
  "tool": "move",
  "status": "completed",
  "requested": {},
  "measurement": {},
  "events": []
}
```

The human-readable result remains in the conversation for model compatibility;
the envelope is also persisted in the episode transcript. The executor reports
measurements and rejects invalid protocol calls, but does not prescribe the
next action.

Use the old surface only for A/B comparison:

```bash
bash scripts/agentenv.sh -m arm_agent.cli run --tool-protocol v1 ...
```
