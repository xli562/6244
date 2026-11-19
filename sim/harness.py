"""Agent harness: let a coding agent (Codex CLI or Claude Code CLI) develop a policy against
a set of simulated development scenes, under a fixed evaluation budget.

Two modes:
  claude - autonomous agent: Claude Code works in the workspace with file tools and one
           allowed command, `python3 evaluate.py`, and decides itself when to evaluate.
  codex  - harness-driven loop without tools: Codex replies with the whole program, the
           harness evaluates it and sends the feedback back (Codex's command sandbox does
           not start on this machine). Same documents, scenes and evaluation budget.

In the autonomous mode the agent works in a workspace directory that holds only the task description, the skill
API documentation, a starter policy and a small `evaluate.py` client. The client drops a
request file; this process (outside the agent's sandbox) runs the simulation and writes
structured feedback back. The simulator code, the judge and the test sets are not in the
workspace.

    python sim/harness.py --agent codex --condition family --run-id pilot_codex_family
"""

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from glob import glob
from pathlib import Path

import episode
import skills
import variants
from repair_cli import AGENTS

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
WORKSPACES = ROOT.parent / "agent_runs"  # outside the repository on purpose

CONDITIONS = {  # condition -> development set
    "oneshot": None,
    "twin": "twin",
    "family": "family",
    "family_v2": "family_v2",
    "naive": "naive",
}

PROMPT = ("Read TASK.md in the current directory and complete the task it describes. "
          "Work only inside the current directory.")

TASK = """\
# Task: put the cup in the drawer

A Franka Panda arm stands in front of a kitchen counter. Write a robot program that
makes it carry out the user's instruction:

> "Place the cup in the drawer."

The task is done when the cup stands upright inside the drawer, the gripper has let go
of it, the drawer is closed, and no other object has been knocked off the counter.

## What to deliver

`policy.py` in this directory, defining `run(api)`. `API.md` documents the `api` object;
it is the only way your program can sense and act. `SCENE.md` describes the reference simulated kitchen.

## How your program will be used

After you finish, `policy.py` is run in held-out simulated kitchens, many times, in situations you
have not seen. Things will not be exactly where they are in your development scenes. It
must work without you: nobody will edit it afterwards.

## Development scenes

{dev_text}

## Rules

- Work only inside this directory. Do not look for the simulator, other scenes or other
  programs elsewhere on this machine.
- `policy.py` may import only `numpy` and `math`.
- Stop when you are satisfied with `policy.py` or the evaluation budget is used up.
"""

DEV_TEXT = {
    None: "There is no simulator access in this run: write the best program you can from "
          "the documents alone.",
    "sim": "You can test `policy.py` in simulation with\n\n"
           "    python3 evaluate.py policy.py\n\n"
           "It runs your program on {n} development scenes and prints what happened in each. "
           "The full record of every run, with the trace of your API calls for failed episodes, "
           "is saved under `feedback/`. You have a budget of **{budget} evaluations**; a file "
           "that fails the import check does not use one up.\n\n{what}",
}
SET_TEXT = {
    "twin": "The development scenes are the reconstructed kitchen (`SCENE.md`) in its "
            "reference layout. The {n} scenes differ only in the perception error.",
    "family": "The development scenes are {n} variations of the reconstructed kitchen "
              "(`SCENE.md`): the cup position and orientation, how far the drawer is open at "
              "the start, and the other objects on the counter and in the drawer differ from "
              "scene to scene.",
    "family_v2": "The development scenes are {n} variations of the reconstructed kitchen "
                 "(`SCENE.md`): the cup position and orientation, how far the drawer is open at "
                 "the start, and the other objects on the counter and in the drawer differ from "
                 "scene to scene. The size of the perception error differs too, from about 2.5 mm "
                 "to 10 mm, because it is not known how accurate perception will be in the test "
                 "kitchen; the evaluation output gives each scene's level.",
    "naive": "The development scenes are {n} randomised variations of the reconstructed "
             "kitchen (`SCENE.md`): the cup position and orientation, how far the drawer is "
             "open at the start, and the other objects on the counter and in the drawer differ "
             "from scene to scene. The variations were not checked, so some may be unsolvable.",
}

STARTER = '''\
"""Robot program for: "Place the cup in the drawer." See TASK.md and API.md."""
import math

import numpy as np


def run(api):
    raise NotImplementedError("write the program here")
'''

CLIENT = '''\
"""Evaluate a policy file on the development scenes.  Usage: python3 evaluate.py policy.py"""
import json
import os
import sys
import time

path = sys.argv[1] if len(sys.argv) > 1 else "policy.py"
here = os.path.dirname(os.path.abspath(__file__))
queue = os.path.join(here, ".eval")
os.makedirs(queue, exist_ok=True)
tag = "%d_%d" % (int(time.time() * 1000), os.getpid())
with open(path) as f:
    source = f.read()
tmp = os.path.join(queue, "tmp_%s.json" % tag)
with open(tmp, "w") as f:
    json.dump({"policy_source": source}, f)
os.rename(tmp, os.path.join(queue, "req_%s.json" % tag))
resp = os.path.join(queue, "resp_%s.json" % tag)
deadline = time.time() + 900
while not os.path.exists(resp):
    if time.time() > deadline:
        sys.exit("evaluation service did not answer within 15 minutes")
    time.sleep(0.5)
time.sleep(0.2)
with open(resp) as f:
    print(json.load(f)["text"])
'''


def scene_doc(cfg):
    hx, hy, hz = cfg.interior_half
    handle_x = cfg.counter_front_x + cfg.handle_offset_x
    return f"""\
# The reference simulated kitchen

These nominal values come from product dimensions and modelling assumptions. The test
kitchens can differ from this development configuration. Frame and units as in `API.md`.

- Counter: top surface at z = {cfg.counter_top_z:.3f}; front face at x = {cfg.counter_front_x:.3f};
  it spans y from {-cfg.counter_half_y:.2f} to {cfg.counter_half_y:.2f}.
- Drawer: in the counter front, centred at y = {cfg.drawer_y:.3f}. It slides out toward the
  robot (-x) by at most {cfg.drawer_max_open:.2f}. It starts closed in this layout.
- Drawer interior: {2 * hx:.3f} long (x), {2 * hy:.3f} wide (y), side walls {cfg.drawer_wall_height:.3f} high;
  the drawer floor is at z = {cfg.drawer_z + 2 * cfg.drawer_wall:.3f}. The drawer front panel is
  {cfg.drawer_front_height:.3f} high and {cfg.drawer_panel:.3f} thick.
- Handle: a vertical bar, 0.016 thick and {cfg.handle_z[1] - cfg.handle_z[0]:.3f} long, standing
  {-cfg.handle_offset_x:.3f} in front of the drawer front. With the drawer closed its grasp point is
  at ({handle_x:.3f}, {cfg.drawer_y:.3f}, {cfg.drawer_z + cfg.handle_z[1] - 0.03:.3f}).
- Cup: an espresso cup with a handle, body diameter {cfg.cup_diameter:.3f}, height {cfg.cup_height:.3f},
  about {cfg.cup_mass:.2f} kg. In the reference layout it stands on the counter at
  ({cfg.cup_xy[0]:.2f}, {cfg.cup_xy[1]:.2f}) with its handle pointing toward the robot.
- Robot: Franka Panda, base at the origin, parallel gripper that opens to 0.08.
"""


def describe_scene(ep):
    s = ep["scene"]
    on_counter = sum(c["where"] == "counter" for c in s["clutter"])
    in_drawer = len(s["clutter"]) - on_counter
    text = (f"cup at ({s['cup_xy'][0]:.2f}, {s['cup_xy'][1]:.2f}) yaw {s['cup_yaw']:+.2f}, "
            f"drawer initially open {s['drawer_open']:.2f}, "
            f"{on_counter} other objects on the counter, {in_drawer} in the drawer")
    if ep.get("noise_shown"):
        text += f", perception error level {1000 * ep['noise_sigma']:.1f} mm"
    return text


def feedback_text(index, budget, episodes, results, json_name):
    summary = episode.summarize(results)
    lines = [f"Evaluation {index} of {budget}: {summary['successes']}/{summary['episodes']} "
             f"development scenes succeeded.",
             "Scenes that got at least as far as each stage: "
             + ", ".join(f"{k} {v}" for k, v in summary["reached_stage"].items() if k != "start"),
             ""]
    for i, (ep, r) in enumerate(zip(episodes, results)):
        head = f"scene {i} [{describe_scene(ep)}]: "
        if r["success"]:
            line = head + "SUCCESS"
        else:
            f = r["final"]
            line = head + f"FAILED - {r['failure']}"
            if f:
                line += (f"; cup ended in/on: {f['cup_location']}, tilt {f['cup_tilt_deg']} deg, "
                         f"drawer opening at the end {f['drawer_opening']}")
        ev = r.get("events") or {}
        if ev.get("collision_seconds"):
            line += "; collisions (seconds in contact): " + json.dumps(ev["collision_seconds"])
        if ev.get("objects_moved_m"):
            line += "; objects pushed (m): " + json.dumps(ev["objects_moved_m"])
        if r.get("error"):
            line += "\n    error: " + r["error"].splitlines()[-1]
        lines.append(line)
    lines += ["", f"Full record (checks, final state, API-call trace of failed scenes): feedback/{json_name}",
              f"Evaluations left: {budget - index}"]
    return "\n".join(lines)


class EvalServer(threading.Thread):
    """Answers evaluation requests dropped into <workspace>/.eval by evaluate.py."""

    def __init__(self, workspace, run_dir, episodes, budget):
        super().__init__(daemon=True)
        self.workspace, self.run_dir = Path(workspace), Path(run_dir)
        self.episodes, self.budget = episodes, budget
        self.queue = self.workspace / ".eval"
        self.queue.mkdir(exist_ok=True)
        (self.workspace / "feedback").mkdir(exist_ok=True)
        self.used = 0
        self.history = []
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            for req in sorted(self.queue.glob("req_*.json")):
                tag = req.stem[len("req_"):]
                try:
                    source = json.loads(req.read_text())["policy_source"]
                    text = self.handle(source)
                except Exception as e:  # noqa: BLE001
                    text = f"evaluation service error: {type(e).__name__}: {e}"
                req.unlink(missing_ok=True)
                tmp = self.queue / f"tmp_resp_{tag}.json"
                tmp.write_text(json.dumps({"text": text}))
                tmp.rename(self.queue / f"resp_{tag}.json")
            time.sleep(0.3)

    def handle(self, source):
        if not self.episodes:
            return "This run has no simulator access: no evaluations are available."
        if self.used >= self.budget:
            return f"Evaluation budget used up ({self.budget} of {self.budget}). No evaluation was run."
        try:
            episode.check_policy_source(source)
        except episode.PolicyError as e:
            return f"policy.py was rejected before running (this did not use up an evaluation): {e}"
        self.used += 1
        index = self.used
        started = time.time()
        results = episode.run_episodes(self.episodes, source, trace_limit=80)
        summary = episode.summarize(results)
        out = self.run_dir / "evals" / f"eval_{index:02d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "policy.py").write_text(source)
        (out / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1))
        self.history.append({"index": index, "successes": summary["successes"],
                             "episodes": summary["episodes"], "failures": summary["failures"],
                             "seconds": round(time.time() - started, 1)})
        json_name = f"eval_{index:02d}.json"
        shown = [{**r, "scene": describe_scene(ep), "trace": [] if r["success"] else r["trace"]}
                 for ep, r in zip(self.episodes, results)]
        (self.workspace / "feedback" / json_name).write_text(json.dumps(shown, indent=1))
        text = feedback_text(index, self.budget, self.episodes, results, json_name)
        (self.workspace / "feedback" / f"eval_{index:02d}.txt").write_text(text)
        return text


def find_binary(agent):
    paths = sorted(glob(os.path.expanduser(AGENTS[agent]["glob"])))
    if not paths and shutil.which(agent):
        return shutil.which(agent)
    if not paths:
        raise FileNotFoundError(f"no {agent} binary matches {AGENTS[agent]['glob']}")
    return paths[-1]


def agent_command(agent, workspace):
    spec, binary = AGENTS[agent], find_binary(agent)
    if agent == "codex":
        return [binary, "exec", "-m", spec["model"], "-c", f"model_reasoning_effort={spec['effort']}",
                "-s", "workspace-write", "--skip-git-repo-check", "--ephemeral", "--json",
                "-C", str(workspace), PROMPT]
    return [binary, "-p", PROMPT, "--model", spec["model"], "--effort", spec["effort"],
            "--output-format", "stream-json", "--verbose", "--permission-mode", "acceptEdits",
            "--tools", "Read,Edit,Write,Glob,Grep,Bash",
            "--allowedTools", "Read", "Edit", "Write", "Glob", "Grep", "Bash(python3 evaluate.py:*)",
            "Bash(ls:*)", "Bash(cat:*)", "--strict-mcp-config", "--setting-sources", ""]


def parse_log(agent, log_path, workspace):
    """Token usage and an audit of commands that reach outside the workspace."""
    usage, commands, final_text = {}, [], None
    for line in Path(log_path).read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if agent == "codex":
            if ev.get("type") == "turn.completed":
                for k, v in (ev.get("usage") or {}).items():
                    usage[k] = usage.get(k, 0) + v
            item = ev.get("item") or {}
            if ev.get("type") == "item.completed":
                if item.get("type") == "command_execution":
                    commands.append(item.get("command", ""))
                elif item.get("type") == "agent_message":
                    final_text = item.get("text")
        else:
            if ev.get("type") == "result":
                usage = {**(ev.get("usage") or {}), "total_cost_usd": ev.get("total_cost_usd"),
                         "num_turns": ev.get("num_turns"), "is_error": ev.get("is_error")}
                final_text = ev.get("result")
            message = ev.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            for block in (content if isinstance(content, list) else []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    commands.append(json.dumps(block.get("input", {})))
    usage = {k: v for k, v in usage.items() if isinstance(v, (int, float, bool))}
    outside = [c[:200] for c in commands
               if str(ROOT) in c or "/sim/" in c or "../" in c or "outputs/sets" in c]
    return usage, len(commands), outside, final_text


def finalize(agent, condition, run_id, budget=None, wall_seconds=None, timed_out=None,
             exit_code=None):
    """Write runs/<run_id>/run.json from what the run left on disk (also usable after a crash)."""
    run_dir, workspace = RUNS_DIR / run_id, WORKSPACES / run_id
    final = workspace / "policy.py"
    final_source = final.read_text() if final.exists() else ""
    (run_dir / "final_policy.py").write_text(final_source)
    try:
        episode.check_policy_source(final_source)
        valid, problem = "NotImplementedError" not in final_source, None
    except episode.PolicyError as e:
        valid, problem = False, str(e)
    history = []
    for path in sorted((run_dir / "evals").glob("eval_*/results.json")):
        summary = json.loads(path.read_text())["summary"]
        history.append({"index": int(path.parent.name[-2:]), "successes": summary["successes"],
                        "episodes": summary["episodes"], "failures": summary["failures"]})
    usage, n_commands, outside, final_text = parse_log(agent, run_dir / "agent_log.jsonl", workspace)
    set_name = CONDITIONS[condition]
    meta = {"run_id": run_id, "agent": agent, **{k: AGENTS[agent][k] for k in ("model", "effort")},
            "condition": condition, "dev_set": set_name,
            "dev_episodes": history[0]["episodes"] if history else 0,
            "budget": budget, "evaluations_used": len(history), "evaluation_history": history,
            "wall_seconds": wall_seconds, "timed_out": timed_out, "exit_code": exit_code,
            "final_policy_valid": valid, "final_policy_problem": problem,
            "usage": usage, "agent_commands": n_commands, "outside_workspace_access": outside,
            "final_message": (final_text or "")[:1500]}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    return meta


def dev_episodes(set_name, n_dev, dev_set=None, dev_offset=0):
    """The run's development scenes: a slice of the condition's set (or of `dev_set`)."""
    if not set_name:
        return []
    episodes = variants.load_set(dev_set or set_name)[dev_offset:dev_offset + n_dev]
    if len(episodes) < n_dev:
        raise ValueError(f"{dev_set or set_name} has no {n_dev} scenes from offset {dev_offset}")
    return episodes


def run(agent, condition, run_id, n_dev=8, budget=5, timeout=1500, dev_set=None, dev_offset=0,
        log=lambda *a: print(*a, flush=True)):
    run_dir, workspace = RUNS_DIR / run_id, WORKSPACES / run_id
    for d in (run_dir, workspace):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    set_name = CONDITIONS[condition]
    episodes = dev_episodes(set_name, n_dev, dev_set, dev_offset)
    if set_name:
        dev_text = DEV_TEXT["sim"].format(n=len(episodes), budget=budget,
                                          what=SET_TEXT[set_name].format(n=len(episodes)))
    else:
        dev_text = DEV_TEXT[None]
    (workspace / "TASK.md").write_text(TASK.format(dev_text=dev_text))
    (workspace / "API.md").write_text(skills.API_DOC)
    (workspace / "SCENE.md").write_text(scene_doc(variants.make_twin()))
    (workspace / "policy.py").write_text(STARTER)
    (workspace / "evaluate.py").write_text(CLIENT)
    for name in ("TASK.md", "API.md", "SCENE.md"):
        shutil.copy2(workspace / name, run_dir / name)

    server = EvalServer(workspace, run_dir, episodes, budget)
    server.start()
    command = agent_command(agent, workspace)
    log(f"[{run_id}] starting {agent} ({AGENTS[agent]['model']}, {AGENTS[agent]['effort']}) "
        f"on condition '{condition}', {len(episodes)} dev scenes, budget {budget}")
    started = time.time()
    log_path = run_dir / "agent_log.jsonl"
    timed_out = False
    with open(log_path, "w") as out, open(run_dir / "agent_stderr.txt", "w") as err:
        proc = subprocess.Popen(command, cwd=workspace, stdout=out, stderr=err,
                                stdin=subprocess.DEVNULL)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
    server.stop_flag.set()
    server.join(timeout=5)
    seconds = round(time.time() - started, 1)

    meta = finalize(agent, condition, run_id, budget=budget, wall_seconds=seconds,
                    timed_out=timed_out, exit_code=proc.returncode)
    log(f"[{run_id}] finished in {seconds:.0f} s, exit {proc.returncode}, "
        f"{meta['evaluations_used']}/{budget} evaluations, dev history "
        f"{[str(h['successes']) + '/' + str(h['episodes']) for h in meta['evaluation_history']]}")
    return meta


# --- harness-driven loop (no tools) ---------------------------------------------
# Codex cannot start its command sandbox on this machine, so it is not run as an autonomous
# agent. Instead the harness drives the loop: Codex receives the documents and replies with
# the whole program; the harness evaluates it and sends the feedback back, `budget` times.
# Codex executes nothing and reads nothing, so it needs no sandbox and cannot see anything
# beyond what is in its prompt.

LOOP_RULES = """\
## How this session works

You have no tools in this session: do not try to run commands or read files. Everything
you need is in this message.

- Reply with the complete contents of `policy.py` in one ```python code block. You may
  write a short explanation before it; only the last ```python block in your reply is used.
- `policy.py` may import only `numpy` and `math`.
{loop_text}
"""
LOOP_TEXT = {
    None: "- There is no simulator access in this run. Your first reply is final.",
    "sim": "- After each reply the program is run on the {n} development scenes and you get "
           "the result back, with the API-call trace of some failed scenes. This happens "
           "**{budget} times**. After the last result you send the final program, which is "
           "not evaluated again before it is used.\n- {what}",
}
LOOP_TASK_HEAD = TASK.split("## What to deliver")[0]
LOOP_USE = TASK.split("## How your program will be used")[1].split("## Development scenes")[0]


def loop_first_prompt(set_name, n, budget):
    if set_name:
        loop_text = LOOP_TEXT["sim"].format(
            n=n, budget=budget,
            what=SET_TEXT[set_name].format(n=n).replace("(`SCENE.md`)", "(described below)"))
    else:
        loop_text = LOOP_TEXT[None]
    return "\n".join([
        LOOP_TASK_HEAD.strip(), "",
        "## What to deliver", "",
        "A Python file `policy.py` defining `run(api)`. The `api` object, documented below, is "
        "the only way your program can sense and act.", "",
        "## How your program will be used", LOOP_USE.rstrip(), "",
        LOOP_RULES.format(loop_text=loop_text),
        "---", "", skills.API_DOC, "---", "", scene_doc(variants.make_twin()),
    ])


def format_trace(trace, last=30):
    lines = []
    for c in trace[-last:]:
        args = ", ".join(f"{k}={json.dumps(v)}" for k, v in (c.get("args") or {}).items())
        result = "" if c.get("result") is None else f" -> {json.dumps(c['result'])}"
        lines.append(f"  t={c['t']:6.2f}  {c['call']}({args}){result}")
    return "\n".join(lines)


def loop_feedback(text, results, index, budget, max_traces=3):
    parts = [text.split("\n\nFull record")[0]]
    failed = [(i, r) for i, r in enumerate(results) if not r["success"]]
    seen = set()
    for i, r in failed:
        if r["failure"] in seen or len(seen) >= max_traces:
            continue
        seen.add(r["failure"])
        parts.append(f"\nAPI-call trace of scene {i} ({r['failure']}), last calls:\n"
                     + format_trace(r["trace"]))
        if r.get("error"):
            parts.append("Error:\n" + r["error"])
    left = budget - index
    if left:
        parts.append(f"\nEvaluations left: {left}. Reply with the next version of the complete "
                     "program in one ```python block.")
    else:
        parts.append("\nThat was the last evaluation. Reply with the final program in one "
                     "```python block (it may be any earlier version, unchanged, or a new one; "
                     "it will not be evaluated again before use).")
    return "\n".join(parts)


def extract_program(text):
    import re
    blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text or "", flags=re.S)
    return blocks[-1].strip() + "\n" if blocks else None


def codex_turn(prompt, workspace, log_file, session_id=None, timeout=900):
    """One Codex reply, with no tool use. Returns (reply text, session id)."""
    spec, binary = AGENTS["codex"], find_binary("codex")
    common = ["-m", spec["model"], "-c", f"model_reasoning_effort={spec['effort']}",
              "-c", 'sandbox_mode="read-only"', "--skip-git-repo-check", "--json"]
    if session_id:
        command = [binary, "exec", "resume", *common, session_id, "-"]
    else:
        command = [binary, "exec", *common, "-C", str(workspace), "-"]
    proc = subprocess.run(command, cwd=workspace, input=prompt, capture_output=True, text=True,
                          timeout=timeout)
    log_file.write(proc.stdout)
    log_file.flush()
    reply = None
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "thread.started":
            session_id = session_id or ev.get("thread_id")
        item = ev.get("item") or {}
        if ev.get("type") == "item.completed" and item.get("type") == "agent_message":
            reply = item.get("text")
    if reply is None:
        raise RuntimeError(f"codex gave no reply (exit {proc.returncode}): {proc.stderr[-400:]}")
    return reply, session_id


def run_loop(condition, run_id, n_dev=8, budget=5, timeout=2400, dev_set=None, dev_offset=0,
             log=lambda *a: print(*a, flush=True)):
    """Harness-driven development loop for Codex (see the comment above)."""
    agent = "codex"
    run_dir, workspace = RUNS_DIR / run_id, WORKSPACES / run_id
    for d in (run_dir, workspace):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    set_name = CONDITIONS[condition]
    episodes = dev_episodes(set_name, n_dev, dev_set, dev_offset)
    rounds = budget if set_name else 0
    server = EvalServer(workspace, run_dir, episodes, rounds)  # used directly, not as a thread
    prompt = loop_first_prompt(set_name, len(episodes), rounds)
    (run_dir / "first_prompt.md").write_text(prompt)
    log(f"[{run_id}] starting codex loop ({AGENTS[agent]['model']}, {AGENTS[agent]['effort']}) "
        f"on condition '{condition}', {len(episodes)} dev scenes, {rounds} evaluations")
    started, session, turns, problem = time.time(), None, 0, None
    program = None
    with open(run_dir / "agent_log.jsonl", "w") as log_file, \
            open(run_dir / "prompts.md", "w") as prompts:
        def ask(text):
            nonlocal session, turns
            prompts.write(f"\n\n===== prompt {turns + 1} =====\n\n{text}")
            prompts.flush()
            reply, session = codex_turn(text, workspace, log_file, session)
            turns += 1
            prompts.write(f"\n\n===== reply {turns} =====\n\n{reply}")
            return reply

        def get_program(text):
            """Ask until the reply holds a program that passes the import check (3 tries)."""
            for _ in range(3):
                source = extract_program(ask(text))
                if source is None:
                    text = "Your reply had no ```python code block. Send the complete program in one."
                    continue
                try:
                    episode.check_policy_source(source)
                    return source
                except episode.PolicyError as e:
                    text = (f"The program was rejected before running (this did not use up an "
                            f"evaluation): {e}. Send the corrected complete program.")
            raise RuntimeError("no valid program after 3 attempts")

        try:
            program = get_program(prompt)
            (workspace / "policy.py").write_text(program)
            for index in range(1, rounds + 1):
                if time.time() - started > timeout:
                    problem = "wall-clock limit reached"
                    break
                text = server.handle(program)
                results = json.loads((run_dir / "evals" / f"eval_{index:02d}" / "results.json")
                                     .read_text())["results"]
                h = server.history[-1]
                log(f"[{run_id}] evaluation {index}: {h['successes']}/{h['episodes']}")
                program = get_program(loop_feedback(text, results, index, rounds))
                (workspace / "policy.py").write_text(program)
        except Exception as e:  # noqa: BLE001
            problem = f"{type(e).__name__}: {e}"
            log(f"[{run_id}] stopped: {problem}")
    seconds = round(time.time() - started, 1)
    meta = finalize(agent, condition, run_id, budget=rounds, wall_seconds=seconds,
                    timed_out=problem == "wall-clock limit reached", exit_code=0 if not problem else 1)
    meta.update({"mode": "harness_loop", "agent_turns": turns, "loop_problem": problem})
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    log(f"[{run_id}] finished in {seconds:.0f} s, {turns} Codex turns, dev history "
        f"{[str(h['successes']) + '/' + str(h['episodes']) for h in meta['evaluation_history']]}")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=list(AGENTS), required=True)
    parser.add_argument("--condition", choices=list(CONDITIONS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-dev", type=int, default=8)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=1500, help="wall-clock limit in seconds")
    parser.add_argument("--dev-set", help="scene set to develop on (default: the condition's set)")
    parser.add_argument("--dev-offset", type=int, default=0,
                        help="index of the first development scene in that set")
    parser.add_argument("--select-checkpoint", action="store_true",
                        help="also save selected_policy.py using 16 independent simulated validation scenes")
    parser.add_argument("--finalize-only", action="store_true",
                        help="rebuild run.json for a run that already happened")
    args = parser.parse_args()
    if args.finalize_only:
        print(json.dumps(finalize(args.agent, args.condition, args.run_id, budget=args.budget), indent=1))
        return
    if args.agent == "codex":
        meta = run_loop(args.condition, args.run_id, args.n_dev, args.budget,
                        max(args.timeout, 2400), args.dev_set, args.dev_offset)
    else:
        meta = run(args.agent, args.condition, args.run_id, args.n_dev, args.budget, args.timeout,
                   args.dev_set, args.dev_offset)
    meta.update({"dev_set_file": args.dev_set or CONDITIONS[args.condition],
                 "dev_offset": args.dev_offset})
    (RUNS_DIR / args.run_id / "run.json").write_text(json.dumps(meta, indent=1))
    if args.select_checkpoint and meta.get("evaluations_used"):
        from replay_selection import publish_selection
        publish_selection(RUNS_DIR / args.run_id)


if __name__ == "__main__":
    main()
