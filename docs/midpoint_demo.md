# Midpoint Demo: Cup in Drawer (Simulation)

Midpoint check: Mon 19 Oct 2026. Status: **M1 done** (sim baseline and scripted oracle).

## 1. Show this

1. Video: `sim/outputs/demo_cup_drawer.mp4` (29 s; overview + side camera)
2. The family: `sim/outputs/family/` — 20 standalone scene files (`variant_000.xml` …), shared `assets/`, `manifest.json`, and `grid.png` (8 of the 20 scenes)
3. Numbers: oracle succeeds on the default scene, on **20/20** exported variants, and on **50/50** from a second seed
4. A failure the variants caught: with the cup rotated 0.5 rad, the open gripper snagged the mug handle while lifting away and tipped the cup. Fix: rotate the gripper with the cup. One scene would never have shown this.

## 2. Reproduce (≈2 min)

```bash
conda activate robot          # Python 3.11, MuJoCo 3.16.0, mink 1.3.0
sim/setup_assets.sh           # Panda + mug, pinned commits, into third_party/
python sim/demo_cup_drawer.py # renders the MP4, prints the success check
python sim/variants.py --n 20 # export 20 variants as XML files, verify each with the oracle
python sim/demo_cup_drawer.py --xml sim/outputs/family/variant_006.xml  # run any one file
```

Headless rendering uses EGL (`MUJOCO_GL=egl` is set in the `robot` env).

## 3. What is built

| File | What it does |
| --- | --- |
| `sim/scene.py` | Builds the scene with `MjSpec`: Panda, counter, drawer (slide joint, handle), scanned mug. All layout is parameterised by `SceneConfig`. |
| `sim/demo_cup_drawer.py` | Scripted oracle: open drawer → pick cup → place in drawer → close drawer. mink differential IK drives the arm. Success = cup upright, inside drawer, drawer closed. |
| `sim/variants.py` | Samples cup x/y/yaw and initial drawer opening, exports each variant as a standalone MJCF file (initial state in keyframe `init`), then runs the oracle on the exported files in parallel. |
| `sim/setup_assets.sh` | Fetches third-party assets at pinned commits. |

## 4. Simplifications (be upfront about these)

1. **Cup scaled to 55%.** Every scanned mug is 9–13 cm wide; the Panda gripper opens to 8 cm. The demo cup is about 5 cm wide, 6.3 cm tall, 0.12 kg.
2. **Drawer is hand-built** from boxes, not reconstructed from a scan yet (that is M2).
3. **Oracle uses ground-truth state.** No perception, no noise.
4. **Gripper stiffened 4×** (finger tendon 400 N/m vs Menagerie's 100 N/m) so the grip holds the cup and handle.

## 5. What the 20/20 result means

The oracle is the **solvability checker**, not the experiment. With perfect state it solves every variant. This is the ceiling-effect risk from the research plan: the real comparison (twin-only vs variant family) needs a mid-level skill API and noisy object poses, or both conditions will score 100%.

## 6. Next milestones

| # | Milestone | Dates |
| --- | --- | --- |
| M2 | Real-to-sim twin S0: measured MJCF, shake test, real-vs-sim clearance table (<5 mm error) | 19 Oct – 1 Nov |
| M3 | Variant generator + checkers: ≥200 verified variants, rejection log, coverage report | 26 Oct – 8 Nov |
| M4 | Agent harness: LLM loop on S0 and on the family, every revision logged | 2 – 15 Nov |
| M5 | Evaluation, ablations, write-up | 9 – 23 Nov |

Full plan: [reports/Real to sim drawer demo plan.md](../reports/Real%20to%20sim%20drawer%20demo%20plan.md).

## 7. Open questions for the mentor

1. Final deadline and deliverable format (course doc does not say; plan assumes ~23 Nov).
2. Is a real robot available for the final test, or is the sim-only proxy acceptable?
3. Which real kitchen and cup to measure for the twin (M2)?
