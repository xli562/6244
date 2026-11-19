"""Observe the preview calls of frozen policies on representative paired changes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import experiment as e
from skill_library import SkillLibrary


def observe(job):
    key, task, index, source, expected, direction = job
    calls = []
    preview = SkillLibrary.preview

    def capture(self, step):
        evidence = preview(self, step)
        calls.append({"step": step.copy(), "evidence": evidence})
        return evidence

    with patch.object(SkillLibrary, "preview", capture):
        row = e.episodes((key, task, [index], source))[0]
    return {
        "program": key,
        "task": task,
        "index": index,
        "change": direction,
        "matches_original": all(
            row[k] == expected[k]
            for k in ("success", "settled", "steps", "error", "trace", "bindings")
        ),
        "previews": calls[:12],
        "preview_calls": len(calls),
        "executed_steps": row["steps"],
    }


def main(batch):
    folder = e.OUT if batch == "first" else e.OUT / "second_batch"
    manifest = e.h.read(folder / "manifest.json")
    e.register(manifest)
    results = e.h.read(folder / "results.json")
    jobs = []
    for k, rows in results.items():
        if not k.endswith("/checked"):
            continue
        comparator = {
            (r["task"], r["index"]): r
            for r in results[k.replace("/checked", "/replay")]
        }
        seen = set()
        for r in rows:
            other = comparator[r["task"], r["index"]]
            change = int(r["settled"]) - int(other["settled"])
            group = (r["task"], change)
            if not change or group in seen:
                continue
            seen.add(group)
            jobs.append(
                (
                    k,
                    r["task"],
                    r["index"],
                    (folder / k / "proposal.py").read_text(),
                    r,
                    change,
                )
            )
    with ProcessPoolExecutor(8) as pool:
        diagnostics = list(pool.map(observe, jobs))
    import json

    (folder / "diagnosis.json").write_text(
        json.dumps(diagnostics, indent=1, default=e.json_value)
    )
    print(
        "matched",
        sum(r["matches_original"] for r in diagnostics),
        "/",
        len(diagnostics),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("batch", choices=["first", "second"])
    main(p.parse_args().batch)
