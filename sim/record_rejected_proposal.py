"""Count a rejected one-call CLI proposal as a failed program without asking the model again."""

import argparse
import json
from pathlib import Path
import re

import episode
from multitask import digest, dump


def record(folder):
    policy = folder / "proposal.py"
    if policy.exists():
        return
    calls = [json.loads(line) for line in (folder / "calls.jsonl").read_text().splitlines()]
    if len(calls) != 1:
        raise ValueError("Expected exactly one CLI call.")
    if calls[0].get("timed_out"):
        reason = f"CLI timed out after {calls[0]['timeout']} seconds without a program."
        save_failure(folder, reason, None)
        return
    if calls[0].get("exit_code") != 0:
        raise ValueError("Expected a timed-out call or one completed CLI response.")
    raw = (folder / "reply.log").read_text()
    if calls[0]["agent"] == "claude":
        response = json.loads(raw)["result"]
    else:
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        response = "\n".join(
            e["item"]["text"]
            for e in events
            if e.get("type") == "item.completed" and e["item"]["type"] == "agent_message"
        )
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", response, re.S)
    if not blocks:
        raise ValueError("No fenced program to audit.")
    source = blocks[-1].strip() + "\n"
    try:
        episode.check_policy_source(source)
    except episode.PolicyError as exc:
        reason = str(exc)
    else:
        raise ValueError("The returned program passed validation; it cannot be marked rejected.")
    save_failure(folder, reason, source)


def save_failure(folder, reason, source):
    # Preserve the returned program. A call that timed out has no source to validate.
    policy = folder / "proposal.py"
    dump(
        folder / "rejection.json",
        dict(reason=reason, sha256=digest(source) if source else None, source=source),
    )
    stub = "def run(api):\n    raise ValueError(" + repr("CLI program rejected: " + reason) + ")\n"
    with policy.open("x") as f:
        f.write(stub)
    print(folder, reason, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="+")
    args = parser.parse_args()
    for folder in args.folder:
        record(folder)
