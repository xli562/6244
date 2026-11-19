"""Read-only grounding of an agent-declared role, independent of the task vocabulary."""

import re
import numpy as np

TOL = 0.002  # Treat differences below 2 mm as an unresolved tie.


def check_role(pick, spec, scene, pos):
    if not isinstance(spec, dict) or spec.keys() - {
        "kind",
        "terms",
        "parent",
        "near",
        "axis",
        "rank",
    }:
        raise ValueError("Unknown selector field")
    kind = spec.get("kind")
    if kind not in {"item", "fixture", "region"}:
        raise ValueError("A role must declare one entity kind")
    terms = spec.get("terms", [])
    if not isinstance(terms, (list, tuple)) or not all(
        isinstance(t, str) and t.isalpha() for t in terms
    ):
        raise ValueError("terms must be descriptive words without instance identifiers")
    if not (terms or spec.get("parent") or spec.get("near")):
        raise ValueError("Missing independent description")
    names = [
        name
        for name, info in scene.items()
        if info["kind"] == kind
        and (not spec.get("parent") or info["parent"] == spec["parent"])
        and set(t.lower() for t in terms) <= set(re.findall(r"[a-z]+", name.lower()))
    ]
    names.sort()
    measures = {}
    choices = names
    near, axis, rank = (spec.get(k) for k in ("near", "axis", "rank"))
    if near is not None:
        if near not in scene or axis is not None or rank is not None or near in names:
            raise ValueError(
                "near needs an independent visible anchor, without axis or rank"
            )
        measures = {n: float(np.linalg.norm(pos(n)[:2] - pos(near)[:2])) for n in names}
        if names:
            choices = [n for n in names if measures[n] <= min(measures.values()) + TOL]
    elif axis is not None or rank is not None:
        if axis not in ("x", "y", "z") or rank not in ("min", "median", "max"):
            raise ValueError("Expected axis x/y/z and rank min/median/max")
        measures = {n: float(pos(n)["xyz".index(axis)]) for n in names}
        if names:
            values = sorted(measures.values())
            center = (values[(len(values) - 1) // 2] + values[len(values) // 2]) / 2
            ref = {"min": values[0], "median": center, "max": values[-1]}[rank]
            best = min(abs(v - ref) for v in values)
            choices = [n for n in names if abs(measures[n] - ref) <= best + TOL]
    resolved = choices[0] if len(choices) == 1 else None
    return {
        "pick": pick,
        "resolved": resolved,
        "candidates": names,
        "finalists": choices,
        "measure": {n: round(v, 4) for n, v in measures.items()},
        "status": (
            "unresolved"
            if resolved is None
            else "confirmed"
            if resolved == pick
            else "corrected"
        ),
    }
