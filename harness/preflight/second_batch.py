"""Additional balanced blind batch after first-batch compilation failures, before task selection."""

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import experiment as e

BASE = e.OUT
OUT = BASE / "second_batch"
COMPACT = """Implementation constraints for BOTH methods: aim for 60-90 lines. Use plain str.lower(),
str.replace(), str.split(), list and dict operations to parse text. No regular expressions, no import re,
no other imports besides numpy or math, no getattr/setattr/hasattr, no private attributes. Use only the
literal documented api methods. Prefer a compact strongest-effort implementation over an exhaustive grammar.
No other feedback, no test task descriptions. Return just one complete Python block."""


def main(stage, workers):
    e.lock()
    e.load_programs()
    E = e.h
    OUT.mkdir(exist_ok=True)
    with (OUT / "batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        E.save_lock(
            OUT / "second_freeze.json",
            dict(
                original=E.digest((BASE / "system.json").read_text()),
                runner=E.digest(Path(__file__).read_text()),
                appendix=COMPACT,
            ),
        )
        if stage == "compile":
            if (BASE / "manifest.json").exists():
                raise RuntimeError("Test list already drawn")
            keys = [
                f"{a}/{r}/{b}"
                for a in E.AGENTS
                for b in e.FAMILIES
                for r in range(1, e.REPS + 1)
            ]

            def one(key):
                prompt = (BASE / key / "prompt.txt").read_text() + "\n" + COMPACT
                src = e.previous.generate(key.split("/")[0], OUT / key, prompt)
                return key, E.digest(src)

            with ThreadPoolExecutor(workers) as pool:
                E.save_lock(OUT / "programs.json", dict(pool.map(one, keys)))
        elif stage == "test":
            manifest = E.read(BASE / "manifest.json")
            if E.read(BASE / "selection_lock.json") != e.selected(manifest):
                raise ValueError("Original selection changed")
            if (OUT / "programs.json").stat().st_mtime_ns > (
                BASE / "manifest.json"
            ).stat().st_mtime_ns:
                raise ValueError("Programs must have frozen before task list")
            e.OUT = OUT
            E.save_lock(OUT / "manifest.json", manifest)
            E.save_lock(OUT / "selection_lock.json", e.selected(manifest))
            e.test(workers)
        else:
            import audit

            e.OUT = OUT
            audit.OUT = OUT
            audit.main()


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=["compile", "test", "audit"])
    cli.add_argument("--workers", type=int, default=12)
    args = cli.parse_args()
    main(args.stage, args.workers)
