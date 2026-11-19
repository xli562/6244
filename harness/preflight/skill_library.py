"""Shared physical operations and their receipts. No task or benchmark lookups."""

from copy import deepcopy

import numpy as np
from binding import check_role

ROLES = {
    "place": ("item", "target"),
    "open": ("target",),
    "close": ("target",),
    "stroke": ("target",),
}
GRASPS = ("standard", "rim", "thin_edge")


class SkillLibrary:
    def __init__(self, sensors, checked):
        self._sensors = sensors
        self._checked = checked
        self.events = []
        self.counts = {"acts": 0, "corrected": 0, "blocked": 0}

    def preview(self, step):
        op = step["op"]
        if op not in ROLES:
            raise ValueError(f"Unknown skill {op}")
        if not self._checked:
            return None
        evidence = {}
        catalog = self._sensors.catalog()
        bindings = step.get("expect", {})
        for role in ROLES[op]:
            selector = bindings.get(role)
            try:
                if role == "item" and (
                    not isinstance(selector, dict) or selector.get("kind") != "item"
                ):
                    raise ValueError("The actor of a place must be a movable item")
                if op in ("open", "close") and (
                    not isinstance(selector, dict) or selector.get("kind") != "region"
                ):
                    raise ValueError("The drawer target must be a region")
                evidence[role] = check_role(
                    step.get(role), selector, catalog, self._sensors.pos
                )
            except (ValueError, TypeError) as exc:
                evidence[role] = {
                    "pick": step.get(role),
                    "resolved": None,
                    "status": "unresolved",
                    "reason": str(exc),
                }
        if (
            op == "place"
            and evidence["item"]["resolved"]
            and evidence["item"]["resolved"] == evidence["target"]["resolved"]
        ):
            evidence["target"].update(
                resolved=None, status="unresolved", reason="Source equals destination"
            )
        return evidence

    def act(self, step):
        api = self._sensors
        self.counts["acts"] += 1
        if self.counts["acts"] > 64:
            raise ValueError("Too many skill requests in one episode")
        evidence = self.preview(step)
        roles = ROLES[step["op"]]
        bound = {
            r: evidence[r]["resolved"] if evidence is not None else step.get(r)
            for r in roles
        }
        event = {
            "intent": deepcopy(step),
            "checked": evidence,
            "bound": bound,
            "start": api.remaining(),
        }
        self.events.append(event)
        if evidence is not None and None in bound.values():
            self.counts["blocked"] += 1
            return {
                "executed": False,
                "check": evidence,
                "remaining": api.remaining(),
                "steps_used": 0,
            }
        if evidence and any(e["status"] == "corrected" for e in evidence.values()):
            self.counts["corrected"] += 1
        if step["op"] == "stroke":
            offset, delta = (
                np.asarray(step[k], dtype=float) for k in ("offset", "delta")
            )
            if any(
                v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) > 0.30
                for v in (offset, delta)
            ):
                raise ValueError(
                    "Expected finite world-space stroke vectors up to 30 cm"
                )
            contact = api.pos(bound["target"]) + offset
            api.grip(True)
            for dest, steps in [
                (contact + [0, 0, 0.10], 25),
                (contact, 20),
                (contact + delta, 35),
            ]:
                receipt = api.attempt(
                    {
                        "op": "move",
                        "xyz": dest,
                        "steps": steps,
                        "watch": bound["target"],
                    }
                )
        elif step["op"] != "place":
            receipt = api.attempt({"op": step["op"], **bound})
        else:
            grasps = step.get("grasps", ["standard"])
            if (
                not isinstance(grasps, (list, tuple))
                or not 1 <= len(grasps) <= 3
                or any(g not in GRASPS for g in grasps)
            ):
                raise ValueError("place requires one to three known grasps")
            for grasp in grasps:
                receipt = api.attempt({"op": "place", **bound, "grasp": grasp})
                # Only switch on evidence that the object was not lifted. Do not repeat a carried object.
                if (
                    receipt["max_lift"] is None
                    or receipt["max_lift"] > 0.025
                    or api.remaining() < 60
                ):
                    break
        return dict(
            receipt,
            executed=True,
            check=evidence,
            steps_used=event["start"] - api.remaining(),
        )

    @property
    def instruction(self):
        return self._sensors.instruction

    def catalog(self):
        return self._sensors.catalog()

    def pos(self, name):
        return self._sensors.pos(name)

    def remaining(self):
        return self._sensors.remaining()
