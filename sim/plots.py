"""Figures for the main experiment and the fidelity probes.

    python sim/plots.py --tag main        # writes sim/outputs/<tag>/figs/*.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#a3a29b"
CONDITIONS = {"oneshot": ("No simulator", AQUA), "twin": ("Twin only", BLUE),
              "family": ("Variant family (layout)", ORANGE),
              "family_v2": ("Variant family v2 (layout + perception error)", YELLOW)}
AGENTS = {"claude": "Claude Code (Opus 5.5)", "codex": "Codex (gpt-6-astra)"}
SETS = {"test_real_100": "Hidden real-scene set, first draw",
        "test_real_b_100": "Hidden real-scene set, second draw",
        "test_id_100": "In-distribution test scenes"}
DEVELOPED = ("twin", "family", "family_v2")  # conditions with simulator access


def of_condition(programs, cond, agent=""):
    """Run ids of one condition (and agent); 'family' must not match 'family_v2'."""
    return [p for p in programs if f"{agent}_{cond}_r" in p]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 160,
})


def finish(fig, path, title, note=None, legend_ax=None, ncol=6):
    """Title, legend row, plots, footnote - stacked top to bottom without overlap."""
    bottom = 0.07 + 0.035 * note.count("\n") if note else 0
    fig.tight_layout(rect=[0, bottom, 1, 0.86 if legend_ax is not None else 0.92])
    fig.suptitle(title, x=0.012, y=0.985, ha="left", va="top", fontsize=13, fontweight="bold")
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=ncol,
                   fontsize=9, handlelength=1.2, columnspacing=1.4)
    if note:
        fig.text(0.012, 0.012, note, ha="left", va="bottom", fontsize=8, color=INK2)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


def load_scores(tag):
    rows = json.loads((HERE / "outputs" / tag / "scores.json").read_text())
    cells = {}
    for r in rows:
        cells.setdefault((r["agent"], r["condition"]), []).append(r)
    return rows, cells


def fig_success(tag, out):
    rows, cells = load_scores(tag)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (set_name, set_title) in zip(axes, SETS.items()):
        oracle = cells[("reference", "reference_oracle")][0]["scores"][set_name]["successes"]
        ax.axhline(oracle, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(1.62, oracle + 1.5, f"hand-written reference {oracle}%", ha="right", fontsize=8, color=INK2)
        for a, agent in enumerate(AGENTS):
            for c, (cond, (label, color)) in enumerate(CONDITIONS.items()):
                runs = [r["scores"][set_name]["successes"] for r in cells.get((agent, cond), [])]
                if not runs:
                    continue
                x = a + (c - 1.5) * 0.215
                ax.bar(x, np.mean(runs), width=0.19, color=color, label=label if a == 0 else None,
                       edgecolor=SURFACE, linewidth=2)
                ax.scatter(x + rng.uniform(-0.055, 0.055, len(runs)), runs, s=14, color=INK, alpha=0.75,
                           zorder=3, linewidths=0)
                ax.annotate(f"{np.mean(runs):.0f}%", (x, 0), (0, -5), textcoords="offset points",
                            ha="center", va="top", fontsize=9, fontweight="bold", annotation_clip=False)
                ax.annotate(f"n={len(runs)}", (x, 0), (0, -18), textcoords="offset points",
                            ha="center", va="top", fontsize=7.5, color=INK2, annotation_clip=False)
        ax.set_xticks(range(len(AGENTS)), list(AGENTS.values()))
        ax.tick_params(axis="x", pad=32, length=0)
        ax.set_ylim(0, 104)
        ax.set_xlim(-0.5, 1.65)
        ax.grid(axis="x", visible=False)
        ax.set_title(set_title)
    axes[0].set_ylabel("Success rate on 100 test scenes (%)")
    finish(fig, out / "fig1_success.png",
           "Simulator access matters most; varying perception error in the family gave Claude about "
           "4 points, Codex none on average",
           "Bars: mean over development runs. Dots: individual runs (each scored on 100 scenes). "
           "n = number of development runs. Second draw: sampled after the first had been used to "
           "decide what family v2 varies.\nOne Codex family-v2 run never learned to close the drawer "
           "(about 20% on every set).", legend_ax=axes[0])


FAILURES = [("Drawer not opened", "drawer_not_opened", AQUA), ("Cup not lifted", "cup_not_lifted", BLUE),
            ("Cup not placed in drawer", "cup_not_placed", ORANGE), ("Cup tipped in drawer", "cup_tipped", YELLOW),
            ("Drawer not closed", "drawer_not_closed", MAGENTA), ("Other", None, GRAY)]


def failure_group(label):
    for _, key, _ in FAILURES[:-1]:
        if label.startswith(key):
            return key
    if label.startswith("cup_still_in_gripper"):
        return "cup_not_placed"
    return None


def fig_failures(tag, out):
    rows, cells = load_scores(tag)
    order = [("claude", "oneshot"), ("claude", "twin"), ("claude", "family"), ("claude", "family_v2"),
             ("codex", "twin"), ("codex", "family"), ("codex", "family_v2")]
    order = [k for k in order if k in cells]
    short = {"oneshot": "no simulator", "twin": "twin only", "family": "family", "family_v2": "family v2"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, set_name in zip(axes, ("test_real_b_100", "test_id_100")):
        set_title = SETS[set_name]
        for y, key in enumerate(order):
            group = cells[key]
            counts = {k: 0.0 for _, k, _ in FAILURES}
            for r in group:
                for label, n in r["scores"][set_name]["failures"].items():
                    counts[failure_group(label)] += n / len(group)
            left = 0.0
            for name, k, color in FAILURES:
                v = counts[k]
                ax.barh(y, v, left=left, height=0.56, color=color, edgecolor=SURFACE, linewidth=2,
                        label=name if y == 0 else None)
                if v >= 2.5:
                    ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color=INK)
                left += v
            ax.text(left + 0.6, y, f"{left:.0f}", va="center", fontsize=9, fontweight="bold")
        ax.set_yticks(range(len(order)), [f"{a.capitalize()} · {short[c]}" for a, c in order])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Failed episodes per 100 test scenes (mean over runs)")
        ax.set_title(set_title)
        ax.set_xlim(0, 38)
    axes[0].invert_yaxis()
    finish(fig, out / "fig2_failures.png",
           "Where the programs fail: closing the drawer and grasping the cup account for most losses",
           "Codex without simulator is left out: all 100 episodes fail (the programs stop on their own checks).",
           legend_ax=axes[0])


def fig_dev_curves(tag, out):
    rows, _ = load_scores(tag)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (agent, agent_title) in zip(axes, AGENTS.items()):
        ends = []
        for cond in DEVELOPED:
            label, color = CONDITIONS[cond]
            runs = [r["dev_history"] for r in rows if r["agent"] == agent and r["condition"] == cond]
            if not runs:
                continue
            for h in runs:
                ax.plot(range(1, len(h) + 1), h, color=color, lw=0.7, alpha=0.18)
            mean = [np.mean([h[i] for h in runs if len(h) > i]) for i in range(5)]
            ax.plot(range(1, 6), mean, color=color, lw=2, marker="o", ms=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=label)
            ends.append((mean[-1], f"{label.split(' (')[0]} {mean[-1]:.1f}"))
        ends.sort()
        placed = []
        for y, text in ends:  # keep the end labels apart
            y = max(y, placed[-1] + 0.5) if placed else y
            placed.append(y)
            ax.text(5.12, y, text, color=INK, fontsize=8.5, va="center")
        ax.set_xticks(range(1, 6))
        ax.set_xlim(0.8, 6.9)
        ax.set_ylim(-0.3, 8.4)
        ax.set_xlabel("Evaluation number (budget: 5 per run)")
        ax.set_title(agent_title)
    axes[0].set_ylabel("Development scenes solved (of 8)")
    finish(fig, out / "fig3_dev_curves.png",
           "Development progress within the evaluation budget",
           "Thick line: mean over the runs that used that evaluation. Thin lines: single runs "
           "(Claude decides itself when to evaluate and may stop early; Codex always uses all 5).\n"
           "Family v2 development scenes have perception errors up to 10 mm, so they are harder to solve.",
           legend_ax=axes[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()
    out = HERE / "outputs" / args.tag / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig_success(args.tag, out)
    fig_failures(args.tag, out)
    fig_dev_curves(args.tag, out)
    if (HERE / "outputs" / args.tag / "probes.json").exists():
        import plots_probes
        plots_probes.make(args.tag, out)


if __name__ == "__main__":
    main()
