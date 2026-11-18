"""Figures for the fidelity probes (called from plots.py)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plots as P

HERE = Path(__file__).resolve().parent
ROWS = [  # (condition, label)
    ("geometry_real", "Geometry: all 8 dimensions set to the real values"),
    ("drawer_friction_real", "Drawer friction: real value (x1.7)"),
    ("drawer_friction_high", "Drawer friction: high (x2.5)"),
    ("drawer_damping_low", "Drawer damping: low (x0.4)"),
    ("drawer_damping_high", "Drawer damping: high (x1.15)"),
    ("cup_mass_low", "Cup mass: light (x0.8)"),
    ("cup_mass_high", "Cup mass: heavy (x1.5)"),
    ("cup_friction_low", "Cup friction: slippery (x0.5)"),
    ("physics_real", "Physics: all 4 parameters set to the real values"),
    ("wide_layout", "Layout: wider ranges, more clutter"),
    ("noise_7.5mm", "Perception error: 5 mm -> 7.5 mm"),
    ("all_together", "All together (hidden real-scene set)"),
]
NOISE = [("noise_0mm", 0), ("noise_2.5mm", 2.5), ("base", 5), ("noise_7.5mm", 7.5), ("noise_10mm", 10),
         ("noise_15mm", 15)]


def load(tag):
    d = json.loads((HERE / "outputs" / tag / "probes.json").read_text())
    return d["results"], d["programs"]


def rate(results, cond, programs):
    return 100 * np.mean([r["success"] for p in programs for r in results[cond][p]])


def fig_probes(tag, out):
    results, programs = load(tag)
    groups = {g: P.of_condition(programs, g) for g in P.DEVELOPED}
    groups = {g: progs for g, progs in groups.items()
              if progs and all(p in results[c] for p in progs for c, _ in ROWS)}
    fig, ax = plt.subplots(figsize=(11, 8.2))
    for y, (cond, label) in enumerate(ROWS):
        for k, (g, progs) in enumerate(groups.items()):
            name, color = P.CONDITIONS[g]
            delta = rate(results, cond, progs) - rate(results, "base", progs)
            yy = y + (k - (len(groups) - 1) / 2) * 0.27
            ax.barh(yy, delta, height=0.24, color=color, edgecolor=P.SURFACE, linewidth=1.5,
                    label=f"Developed on: {name[0].lower() + name[1:]}" if y == 0 else None)
            ax.text(delta - 0.35 if delta < 0 else delta + 0.35, yy, f"{delta:+.1f}", va="center",
                    ha="right" if delta < 0 else "left", fontsize=8.5)
    ax.axvline(0, color=P.INK2, lw=1)
    ax.set_yticks(range(len(ROWS)), [label for _, label in ROWS])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-17, 4)
    ax.set_xlabel("Change in success rate against the unchanged twin scenes (percentage points)")
    P.finish(fig, out / "fig4_probes.png",
             "Which fidelity gap costs success: perception error does, geometry and physics do not",
             "One aspect changed at a time on the same 100 twin-based scenes; mean over 16 developed programs "
             "per bar (both agents). Factors are relative to the twin's values.", legend_ax=ax, ncol=1)


def fig_noise(tag, out):
    results, programs = load(tag)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    xs = [x for _, x in NOISE]
    for ax, (agent, agent_title) in zip(axes, P.AGENTS.items()):
        ax.axvspan(4.6, 5.4, color=P.GRID, alpha=0.6, lw=0)
        ax.text(5, 4, "development", ha="center", fontsize=8, color=P.INK2, rotation=90, va="bottom")
        ax.axvspan(7.1, 7.9, color=P.GRID, alpha=0.6, lw=0)
        ax.text(7.5, 4, "hidden real set", ha="center", fontsize=8, color=P.INK2, rotation=90, va="bottom")
        ref = [rate(results, c, ["reference_oracle"]) for c, _ in NOISE]
        ax.plot(xs, ref, color=P.INK2, lw=1.5, ls=(0, (4, 3)), label="Hand-written reference")
        ends = [(ref[-1], "Reference")]
        for g in P.DEVELOPED:
            name, color = P.CONDITIONS[g]
            progs = P.of_condition(programs, g, agent)
            if not progs or not all(p in results[c] for p in progs for c, _ in NOISE):
                continue
            name = name.split(" (")[0]
            ys = [rate(results, c, progs) for c, _ in NOISE]
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, markeredgecolor=P.SURFACE,
                    markeredgewidth=1.5, label=P.CONDITIONS[g][0])
            ends.append((ys[-1], name))
        ends.sort()
        placed = []
        for y, text in ends:  # keep end labels at least 5 points apart
            y = max(y, placed[-1] + 5) if placed else y
            placed.append(y)
            ax.text(15.4, y, text, fontsize=8.5, va="center")
        ax.set_xticks(xs, [f"{x:g}" for x in xs])
        ax.set_xlim(-0.6, 20.5)
        ax.set_ylim(0, 102)
        ax.set_xlabel("Perception error, standard deviation of the position bias (mm)")
        ax.set_title(agent_title)
    axes[0].set_ylabel("Success rate (%)")
    P.finish(fig, out / "fig5_noise.png",
             "Family v2 programs lose less as perception error grows, but are not better at low error",
             "Same 100 twin-based scenes at every level; only the perception error differs. "
             "Mean over 8 developed programs per line.\nTwo family-v2 programs pull their lines down: "
             "one Claude program fails at 0 mm (6%), one Codex program at every level (about 20%).",
             legend_ax=axes[0])


def fig_noise_failures(tag, out):
    results, programs = load(tag)
    shown = [g for g in P.DEVELOPED if P.of_condition(programs, g)
             and all(p in results[c] for p in P.of_condition(programs, g) for c, _ in NOISE)]
    fig, axes = plt.subplots(1, len(shown), figsize=(5.2 * len(shown) + 0.6, 4.8), sharey=True)
    xs = [x for _, x in NOISE]
    for ax, g in zip(axes, shown):
        progs = P.of_condition(programs, g)
        ends = []
        for name, key, color in P.FAILURES[:-1]:
            ys = [np.mean([100 * np.mean([(r["failure"] is not None and not r["success"]
                                           and P.failure_group(r["failure"]) == key)
                                          for r in results[c][p]]) for p in progs]) for c, _ in NOISE]
            ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, markeredgecolor=P.SURFACE,
                    markeredgewidth=1.2, label=name)
            ends.append((ys[-1], name))
        ends.sort()
        placed = []
        for y, text in ends:
            y = max(y, placed[-1] + 1.3) if placed else y
            placed.append(y)
            ax.text(15.4, y, text, fontsize=8.5, va="center")
        ax.set_xticks(xs, [f"{x:g}" for x in xs])
        ax.set_xlim(-0.6, 23.5)
        ax.set_ylim(0, 24)
        ax.set_xlabel("Perception error, standard deviation of the position bias (mm)")
        ax.set_title(f"Developed on: {P.CONDITIONS[g][0].split(' (')[0].lower()}")
    axes[0].set_ylabel("Failed episodes per 100 scenes")
    P.finish(fig, out / "fig6_noise_failures.png",
             "Drawer-closing failures grow fastest with perception error for family programs, slowest for family v2",
             "Mean over 16 developed programs per panel (both agents), same 100 twin-based scenes at every level.\n"
             "Family v2: one Codex program that never closes the drawer adds about 6 per 100 at every level; "
             "one Claude program adds about 6 at 0 mm.",
             legend_ax=axes[0])


def make(tag, out):
    fig_probes(tag, out)
    fig_noise(tag, out)
    fig_noise_failures(tag, out)
