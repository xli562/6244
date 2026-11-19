"""Stateless program proposals from either CLI. Only the controller scores a proposal."""
import json
import os
import shutil
import subprocess
import tempfile
import time
from glob import glob
from pathlib import Path

from episode import check_policy_source


AGENTS = {
    "codex": {"model": "gpt-6-astra", "effort": "medium",
              "glob": "~/.vscode-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex"},
    "claude": {"model": "claude-opus-5-5", "effort": "high",
               "glob": "~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude"},
}

def binary(agent):
    path = shutil.which(agent)
    if path:
        return path
    paths = sorted(glob(os.path.expanduser(AGENTS[agent]["glob"])))
    if not paths:
        raise FileNotFoundError(f"Install {agent} CLI and sign in.")
    return paths[-1]


def propose(agent, model, effort, prompt, folder, timeout=600):
    import re
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prompt.txt").write_text(prompt)
    with tempfile.TemporaryDirectory(prefix="policy-proposal-") as cwd:
        if agent == "claude":
            command = [binary(agent), "-p", "--model", model, "--effort", effort,
                       "--tools", "", "--strict-mcp-config", "--setting-sources", "",
                       "--no-session-persistence", "--output-format", "json"]
        else:
            command = [binary(agent), "exec", "--ignore-user-config", "--ephemeral",
                       "--skip-git-repo-check", "--disable", "shell_tool", "--disable", "plugins",
                       "-c", 'web_search="disabled"', "-s", "read-only", "-m", model,
                       "-c", f'model_reasoning_effort="{effort}"', "--json", "-"]
        started = time.monotonic()
        call = {"agent": agent, "model": model, "effort": effort, "timeout": timeout}
        try:
            result = subprocess.run(command, cwd=cwd, input=prompt, text=True, capture_output=True,
                                    timeout=timeout)
            call["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            call["timed_out"] = True
            raise
        finally:
            call["seconds"] = round(time.monotonic() - started, 2)
            with (folder / "calls.jsonl").open("a") as log:
                log.write(json.dumps(call) + "\n")
    (folder / "reply.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{agent} exited {result.returncode}: "
                           f"{result.stderr[-500:]} {result.stdout[-500:]}")
    if agent == "claude":
        event = json.loads(result.stdout)
        if event.get("is_error"):
            raise RuntimeError(event.get("result", "Claude CLI error"))
        reply = event["result"]
    else:
        events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        items = [e["item"] for e in events if e.get("type") == "item.completed"]
        if any(i["type"] not in ("agent_message", "reasoning") for i in items):
            raise RuntimeError("Unexpected tool use in policy proposal.")
        reply = "\n".join(i["text"] for i in items if i["type"] == "agent_message")
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", reply, re.S)
    if not blocks:
        raise ValueError("Missing Python block in reply.")
    source = blocks[-1].strip() + "\n"
    check_policy_source(source)
    temporary = folder / "proposal.tmp"
    temporary.write_text(source)
    temporary.replace(folder / "proposal.py")
    return source


def repair_prompt(task, source, feedback):
    return (task + "\nWrite the complete run(api) program in one ```python block. "
            "Use only math, numpy and the supplied API. Use no tools. "
            "Identify the earliest failed subgoal, make a targeted repair, and preserve "
            "previously successful actions.\nCurrent program:\n" + source +
            "\nDevelopment feedback:\n" + feedback)

