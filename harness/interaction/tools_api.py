"""Policy-facing perception and action receipts, independent of task and object names."""

import numpy as np
from grasp_skills import GraspSkills
from libero_long import HORIZON

TOOL_DOC = """api.catalog() returns {name: {'kind': 'item'|'fixture'|'region', 'parent': name or None}}.
Use api.pos(name) for live world xyz. Regions can be attached to movable items or fixtures.
api.remaining() returns control steps left. api.attempt(step) runs a dictionary instruction:
 {'op':'place','item':name,'target':name,'grasp':'standard'|'rim'|'thin_edge'}
 {'op':'open'|'close','target':drawer_region}
 {'op':'move','xyz':[x,y,z],'steps':int,'watch':optional_name}
 {'op':'grip','closed':bool,'watch':optional_name}
It returns an observation dict with steps_used, remaining, watched, before_xyz, after_xyz,
delta_xyz, xy_gap (to target for place), and max_lift (largest recorded lift in metres for place,
null when not measured). Unobserved values are None. These are measurements, not success flags.
Invalid commands raise ValueError or TypeError. You may branch in Python on a receipt or use primitive actions.
On reaching the official terminal state or step limit, execution stops, including inside a skill."""


class InteractionSkills(GraspSkills):
    def __init__(self, env, fallback_source):
        super().__init__(env, "full", fallback_source)
        self.receipts = []
        self.tool_calls = {"catalog": 0, "attempt": 0, "remaining": 0}

    def catalog(self):
        self.tool_calls["catalog"] += 1
        result = {}
        for name, state in self._env.env.object_states_dict.items():
            if state.object_state_type == "site":
                kind = "region"
            else:
                kind = "fixture" if state.is_fixture else "item"
            result[name] = {"kind": kind, "parent": getattr(state, "parent_name", None)}
        return result

    def remaining(self):
        self.tool_calls["remaining"] += 1
        return HORIZON - self._steps

    def attempt(self, step):
        self.tool_calls["attempt"] += 1
        if not isinstance(step, dict):
            raise TypeError("Expected a dictionary action")
        op = step.get("op")
        target = step.get("target")
        watch = step.get("watch")
        if op == "place":
            watch = step.get("item")
            if watch not in self._env.env.objects_dict or target not in self._names:
                raise ValueError("Placement requires a perceived item and target")
        elif op in ("open", "close"):
            watch = target
            if target not in self._names:
                raise ValueError("Unknown drawer region")
        elif op == "move":
            xyz = np.asarray(step.get("xyz"), dtype=float)
            steps = step.get("steps", 30)
            if (
                xyz.shape != (3,)
                or not np.isfinite(xyz).all()
                or not isinstance(steps, int)
                or steps < 1
            ):
                raise ValueError("Expected finite xyz and a positive step count")
        elif op == "grip":
            if not isinstance(step.get("closed"), bool):
                raise ValueError("closed must be a boolean")
        else:
            raise ValueError(f"Unknown action {op}")
        if watch is not None and watch not in self._names:
            raise ValueError(f"Unknown observation target {watch}")
        start, trace_start = self._steps, len(self._trace)
        before = self.pos(watch) if watch else None
        try:
            if op == "place":
                self.place_with(watch, target, step.get("grasp"))
            elif op in ("open", "close"):
                self.drawer(target, op)
            elif op == "move":
                self.move(xyz, min(steps, HORIZON))
            else:
                self.grip(step["closed"])
        finally:
            # Terminal actions still leave observable evidence for the development run.
            after = self.pos(watch) if watch else None
            lifts = [e[3] for e in self._trace[trace_start:] if e[2] == "lift_m"]
            receipt = {
                "steps_used": self._steps - start,
                "remaining": HORIZON - self._steps,
                "watched": watch,
                "before_xyz": None,
                "after_xyz": None,
                "delta_xyz": None,
                "xy_gap": None,
                "max_lift": max(lifts, default=None),
            }
            if watch:
                receipt.update(
                    before_xyz=before.round(4).tolist(),
                    after_xyz=after.round(4).tolist(),
                    delta_xyz=(after - before).round(4).tolist(),
                )
            if op == "place":
                receipt["xy_gap"] = round(
                    float(np.linalg.norm(after[:2] - self.pos(target)[:2])), 4
                )
            if len(self.receipts) < 32:
                self.receipts.append(dict(op=op, target=target, **receipt))
        return receipt


class PolicyInterface:
    """Expose only the documented sensors and controls to policy code."""

    allowed = frozenset(
        [
            "instruction",
            "names",
            "pos",
            "eef",
            "move",
            "grip",
            "action",
            "transfer",
            "arrange",
            "relative_target",
            "drawer",
            "nearest",
            "fallback",
            "place_with",
            "catalog",
            "remaining",
            "attempt",
        ]
    )

    def __init__(self, skills):
        self._skills = skills

    def __getattr__(self, name):
        if name not in self.allowed:
            raise AttributeError(f"Unknown policy API: {name}")
        return getattr(self._skills, name)
