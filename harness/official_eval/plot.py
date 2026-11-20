"""Per-task official success bars with an official − stable deviation strip.

    python harness/official_eval/plot.py --src runs/official_eval_01/test_50_fixed.jsonl.gz --tag _fixed
    python harness/official_eval/plot.py --suite unseen --src runs/unseen_eval_01/test_50_fixed.jsonl.gz --tag _fixed
"""
import argparse, csv, gzip, json, re
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_OUT = ROOT / "runs/official_eval_01"
UNSEEN_OUT = ROOT / "runs/unseen_eval_01"
SUITES = [("10", "LIBERO-10 (long-horizon)"), ("goal", "LIBERO-Goal"),
          ("object", "LIBERO-Object"), ("spatial", "LIBERO-Spatial")]
OFFICIAL, STABLE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated pair)
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
TITLE = "Newest harness (grasp preview + release review)"


def short(task):
    tag, name = task.split("__", 1)
    name = re.sub(r"^(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_", "", name)
    name = re.sub(r"^pick_up_the_black_bowl_", "bowl ", name) if tag == "spatial" else name
    name = (name.replace("pick_up_the_", "") if tag == "object" else name).replace("_and_place_it_in_the_basket", " → basket")
    name = name.replace("_", " ")
    return name if len(name) <= 44 else name[:42] + "…"


def load(src):
    agg = defaultdict(lambda: [0, 0, 0])
    for line in gzip.open(src, "rt"):
        r = json.loads(line)
        a = agg[r["task"]]
        a[0] += 1; a[1] += r["official"]; a[2] += r["stable"]
    return [dict(task=t, n=n, official=o / n, stable=s / n) for t, (n, o, s) in agg.items()]


def mean(rows, key):
    return sum(r[key] * r["n"] for r in rows) / sum(r["n"] for r in rows) * 100


def style():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": INK2, "text.color": INK})


def panel(ax, dx, sub, labels, top, title, xlabel=True, ref=None):
    """Official-success bars on ax; official − stable strip on dx (shared scale `top`)."""
    dev_max = top * 1.15 if top > 0 else 1.0        # strip axis spans the largest observed deviation
    ys = list(range(len(sub)))
    for a in (ax, dx):
        a.set_facecolor(SURFACE)
        a.xaxis.grid(True, color=GRID, lw=.6); a.set_axisbelow(True)
        for s in ("top", "right"): a.spines[s].set_visible(False)
        a.set_ylim(-.6, len(sub) - .4)
    ax.barh(ys, [r["official"] * 100 for r in sub], .62, color=OFFICIAL)
    for y, r in zip(ys, sub):
        ax.text(r["official"] * 100 + 1, y, f"{r['official']*100:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys, labels, fontsize=8.5)
    ax.set_xlim(0, 108); ax.set_xticks(range(0, 101, 25))
    if ref is not None:
        value, text = ref
        ax.axvline(value, color=INK2, lw=1, ls=(0, (4, 3)), zorder=3)
        ax.text(value + 1, len(sub) - .45, text, fontsize=8.5, color=INK2, va="bottom")
    d = [(r["official"] - r["stable"]) * 100 for r in sub]
    dx.hlines(ys, 0, d, color=STABLE, lw=2)
    dx.scatter([v for v in d if v > 0], [y for y, v in zip(ys, d) if v > 0], s=26, color=STABLE, zorder=3)
    for y, v in zip(ys, d):
        if v > 0:
            dx.text(v + dev_max * .06, y, f"{v:.1f}", va="center", fontsize=7.5, color=INK2)
    dx.set_xlim(0, dev_max * 1.45); dx.set_yticks(ys, [""] * len(ys)); dx.tick_params(axis="y", length=0)
    dx.set_xticks([0, round(top, 2)] if top > 0 else [0])
    dx.spines["left"].set_color(MUTED)
    mo, ms = mean(sub, "official"), mean(sub, "stable")
    ax.set_title(f"{title}   mean {mo:.1f}%", loc="left", fontsize=11.5, fontweight="bold")
    dx.set_title(f"−{mo - ms:.2f} pp" if mo > ms else "0 pp", loc="left", fontsize=9, color=INK2)
    if xlabel:
        ax.set_xlabel("Official success rate (%)")
        dx.set_xlabel("Official − stable (pp)", fontsize=8.5)


def header(fig, title, lines, y0=.925, panels="same scale in every panel"):
    fig.suptitle(title, x=.01, ha="left", fontsize=15, fontweight="bold")
    lines = [lines] if isinstance(lines, str) else lines
    lines = lines + ["Bars: official LIBERO success.  Orange strip: drop after the release check "
                     f"(official − stable, percentage points; {panels})."]
    for i, line in enumerate(lines):
        fig.text(.01, y0 - .03 * i, line, fontsize=10, color=INK2)


def save(fig, out, suffix):
    for ext in ("png", "svg"):
        fig.savefig(out / f"success_by_task{suffix}.{ext}", dpi=200, facecolor=SURFACE)
    print("saved", out / f"success_by_task{suffix}.png")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def official(src, suffix=""):
    rows = load(src)
    for r in rows:
        r["suite"] = r["task"].split("__")[0]
    write_csv(OFFICIAL_OUT / f"by_task{suffix}.csv", sorted(rows, key=lambda r: (r["suite"], r["task"])))
    style()
    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    outer = fig.add_gridspec(2, 2, hspace=.32, wspace=.7)
    top = max((r["official"] - r["stable"]) * 100 for r in rows)
    for cell, (tag, title) in zip(outer, SUITES):
        inner = cell.subgridspec(1, 2, width_ratios=(5, 1), wspace=.06)
        sub = sorted([r for r in rows if r["suite"] == tag], key=lambda r: r["official"])
        panel(fig.add_subplot(inner[0]), fig.add_subplot(inner[1]), sub, [short(r["task"]) for r in sub],
              top, title, xlabel=tag in ("object", "spatial"))
    total = sum(r["n"] for r in rows)
    header(fig, TITLE + " on official LIBERO tasks",
           f"Per task: 6 frozen programs (Codex ×3, Claude ×3) × {total // len(rows) // 6} official initial states = "
           f"{total // len(rows)} episodes; {total:,} episodes total. No model calls at test time."
           + (" Non-placement receipts hidden from stage programs (crash fix)." if suffix else ""))
    fig.subplots_adjust(left=.215, right=.965, top=.83, bottom=.07)
    save(fig, OFFICIAL_OUT, suffix)


def unseen(src, suffix=""):
    rows = load(src)
    meta = {t["task"]: t for t in json.loads((UNSEEN_OUT / "manifest.json").read_text())["tasks"]}
    for r in rows:
        m = meta[r["task"]]
        r["instruction"] = m["instruction"]
        r["scene"] = m["scene"]
    write_csv(UNSEEN_OUT / f"by_task{suffix}.csv", sorted(rows, key=lambda r: r["instruction"]))
    ref_src = OFFICIAL_OUT / f"test_50{suffix}.jsonl.gz"
    ref = (mean(load(ref_src), "official"), "official LIBERO mean") if ref_src.exists() else None
    if ref:
        ref = (ref[0], f"official LIBERO mean {ref[0]:.1f}%")
    style()
    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 2, width_ratios=(5, 1), wspace=.04)
    sub = sorted(rows, key=lambda r: r["official"])

    def scene(path):
        name = path.split("/")[-1][:-5]
        m = re.match(r"(KITCHEN|LIVING_ROOM)_SCENE(\d+)_", name)
        where = f"{m[1].replace('_', ' ').lower()} {m[2]}" if m else path.split("/")[0].replace("libero_", "") + " suite"
        return where

    labels = [f"{r['instruction']}   ({scene(r['scene'])})" for r in sub]
    top = max((r["official"] - r["stable"]) * 100 for r in rows)
    panel(fig.add_subplot(grid[0]), fig.add_subplot(grid[1]), sub, labels, top,
          f"{len(rows)} unseen tasks", ref=ref)
    total = sum(r["n"] for r in rows)
    header(fig, TITLE + " on unseen tasks",
           ["New object → destination goals, new instructions and new starting states in 6 stock LIBERO scenes; "
            "none appear anywhere in the project's history.",
            f"Per task: 6 frozen programs (Codex ×3, Claude ×3) × {total // len(rows) // 6} new initial states = "
            f"{total // len(rows)} episodes; {total:,} episodes total. No model calls at test time."],
           panels="axis spans the largest observed drop")
    fig.subplots_adjust(left=.3, right=.965, top=.8, bottom=.08)
    save(fig, UNSEEN_OUT, suffix)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--suite", choices=("official", "unseen"), default="official")
    p.add_argument("--src", type=Path)
    p.add_argument("--tag", default="", help="suffix for output files, e.g. _fixed")
    a = p.parse_args()
    if a.suite == "official":
        official(a.src or OFFICIAL_OUT / "test_50.jsonl.gz", a.tag)
    else:
        unseen(a.src or UNSEEN_OUT / "test_50_fixed.jsonl.gz", a.tag)
