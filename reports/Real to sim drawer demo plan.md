# Develop on Cousins, Test on Unseen Kitchens

**Bottom line:** Build the demo in plain MuJoCo 3.16 inside the existing `robot` env. Use a Menagerie Franka Panda, mink for IK, a hand-written MJCF drawer made of box primitives, and a Google Scanned Objects mug. Do **not** install robosuite into `robot`: a local dry-run on 16 Oct showed it would downgrade numpy from 2.4.6 to 1.26.4 and pin mink to 0.0.5. With this stack, a scripted cup-in-drawer video can exist before the Monday 19 Oct midpoint check. The research pipeline then has four steps:

1. Rebuild the real kitchen as a measured twin.
2. Generate verified variants with MuJoCo's `MjSpec` along three axes: cup pose, drawer opening, clutter.
3. Run one fixed LLM develop→simulate→evaluate→revise loop, once with the twin only and once with the variant family, at equal budget.
4. Score both resulting programs on held-out and "proxy-real" scenes.

The strongest prior evidence favors the hypothesis. On a real cabinet, a single-twin policy succeeded 25% of the time and a "digital cousin" policy 90–95% ([ACDC](https://arxiv.org/html/2410.07408)). As far as the notes found, nobody has run this comparison for an *LLM-written program* rather than a learned policy, so the result would be new. The biggest design risk is not reconstruction. It is a **ceiling effect**: if the skill API is too high-level or perception is noise-free, both conditions reach 100% and the experiment shows nothing. The final deadline is not stated in the course doc. Eight weeks from the 28 Sep proposal suggests about **23 Nov 2026**; confirm this with the mentor.

## Next actions (do these first, in order)

1. **Set up the environment (≈30 min, today).**
   1. `conda activate robot`
   2. `pip install --dry-run mink mediapy` and confirm numpy stays at 2.4.6.
   3. Install mink and mediapy, then `conda install -c conda-forge ffmpeg`.
   4. Clone `google-deepmind/mujoco_menagerie` and sparse-checkout 3–4 mugs from `kevinzakka/mujoco_scanned_objects`.
2. **Write the v0 scene, `kitchen_v0.xml` (≈2 h).**
   1. Counter and cabinet as static boxes.
   2. A drawer body with a `slide` joint, `range`, `damping` and a handle geom. Build its bottom and four walls from separate boxes.
   3. Include the Panda `scene.xml` and the `Cole_Hardware_Mug_Classic_Blue` mug with a free joint.
   4. Shake-test it: step 1 s with zero control and check nothing moves.
3. **Script pick-and-place into a pre-opened drawer (≈3 h).** Use mink IK targets on sites and a waypoint state machine: approach → grasp → lift → over drawer → lower → release. Render a 30 fps MP4 offscreen with `MUJOCO_GL=egl`.
4. **Add drawer opening and closing (≈2–3 h).** Grasp the handle and pull along the slide axis, updating the IK target from the drawer joint `qpos`. You now have the **oracle** that every later phase reuses.
5. **Prepare the midpoint pack (≈1 h, Sun 18 Oct).** Include the MP4, a one-slide pipeline diagram, the five milestones below, and the risk list. Also schedule a 30-min tape-measure session at the real kitchen for the drawer's inner width, depth and height, its travel, and the cup's diameter, height and mass.

## The research proposal (paste-ready, course structure)

### Objectives and motivation

**Problem.** Coding agents can now iterate on robot programs in simulation ([CaP-X](https://proceedings.mlr.press/v306/fu26u.html), [ASPIRE](https://research.nvidia.com/labs/gear/aspire/), [GaP](https://arxiv.org/html/2607.05369v1)). But a program iterated in a single reconstructed scene overfits that scene. The project asks whether **one captured real scene can be turned into a family of task-preserving interactive variants**, so that programs developed across the family generalize to configurations they never saw.

**Input.** One real kitchen scene, captured as follows:

- a phone LiDAR scan (Polycam or Scaniverse)
- photos of the cup
- tape and caliper measurements of the drawer and cup
- the instruction "place the cup in the drawer"

**Output.** Three artifacts:

1. A MuJoCo twin *S0*: an articulated drawer, a cup with real contact, a Franka Panda, and preserved clearances.
2. A generator that emits *K* variants. Each variant is checked as solvable, and cup pose, drawer opening and clutter vary across the family.
3. An evaluation harness. It shows whether an LLM agent's program developed on the family beats one developed on *S0* alone, on held-out scenes.

**Why it matters.** Digital-cousin results show the effect for learned policies: twin-only reached 25% real success, twin plus cousins 95% ([ACDC](https://arxiv.org/html/2410.07408)). Targeted randomization roughly doubled cabinet success over untargeted randomization, 39/50 vs 9/50 ([URDFormer](https://arxiv.org/html/2405.11656v3)). Nobody has tested whether the same holds when the "policy" is code written by an agent. That makes the project a direct contribution to the course question about **verification and agentic feedback**.

**Challenges.** Five stand out:

1. **Concave contact.** MuJoCo collides only with convex hulls of meshes, so a drawer interior or a cup cavity silently "fills in" ([MuJoCo XML reference](https://github.com/google-deepmind/mujoco/blob/main/doc/XMLreference.rst)).
2. **Task preservation.** Raw generated layouts are mostly infeasible: 22.7% collide and 87.3% are unstable before any constraints ([Scenethesis](https://arxiv.org/html/2505.02836v1)).
3. **Agent variance and reward hacking.** Single coding agents can invent trivial success metrics ([GaP](https://arxiv.org/pdf/2607.05369)).
4. **Ceiling effects** when primitives are too powerful.
5. **Limited or no real-robot access**, which forces a sim-only proxy for "unseen real".

### Plan

**Main idea.** The project combines three parts:

1. **Real-to-sim reconstruction.** A measured, primitive-based MJCF twin, rather than a heavy learned reconstruction.
2. **Variant generation.** Constraint-checked sampling with `MjSpec` (the "relation checker + sampler" pattern from SetItUp, without its diffusion models).
3. **A fixed agent loop.** An LLM writes a program against a mid-level skill API and receives structured simulation feedback. Only the development distribution changes between conditions.

**Milestones (overlapping, about 1–2 weeks each):**

| # | Milestone | Done when | Dates |
|---|---|---|---|
| M1 | Sim baseline and scripted oracle | Panda opens drawer, places mug, closes drawer; MP4 rendered | 16–23 Oct |
| M2 | Real-to-sim twin *S0* | Measured MJCF twin passes shake test; real-vs-sim clearance table with error under 5 mm | 19 Oct–1 Nov |
| M3 | Variant generator and checkers | ≥200 verified variants; rejection log; coverage report | 26 Oct–8 Nov |
| M4 | Agent harness | LLM loop runs end to end on *S0* and on the family; every revision logged | 2–15 Nov |
| M5 | Evaluation, ablations, write-up | All conditions × ≥3 agent runs scored on held-out sets; final report and demo video | 9–23 Nov |

### Deliverables

The project delivers four things:

1. A pipeline: capture → twin → variants.
2. A variant library with each variant's parameters stored as JSON.
3. The agent harness and its logs.
4. A demo video of the same agent program succeeding on unseen variants where the twin-only program fails.

**Evaluation.** The **primary metric** is held-out success rate of the final program. Success means three things:

- the cup's centre of mass is inside the drawer's interior box
- the cup is upright
- the gripper has released it

Results are reported for a variant-family condition and a twin-only condition at equal agent budget. Each comes with Wilson 95% CIs and a paired test on identical test configurations. The secondary metrics are:

- the generalization gap
- per-stage progress
- collisions
- development cost in iterations and tokens
- per-axis ablations

## Phased build plan with tools and time

### Phase 0, Fri 16 – Mon 19 Oct: running demo (≈10–14 h of work)

| Component | Choice | Why |
|---|---|---|
| Simulator | Plain MuJoCo 3.16 + Python, CPU physics, EGL rendering | No CUDA/PyTorch dependency, so no Blackwell (sm_120) risk |
| Robot | Menagerie `franka_emika_panda` via `scene.xml` | One-actuator gripper is easiest to script; mink ships `arm_panda.py` ([mink](https://github.com/kevinzakka/mink)) |
| IK | mink 1.3.0 (latest on PyPI), with geom-pair collision avoidance | Avoids robosuite's `mink==0.0.5` pin ([PyPI robosuite](https://pypi.org/pypi/robosuite/json)) |
| Cup | GSO mug from `mujoco_scanned_objects` (1,030 MJCF objects, 32 convex collision pieces each) | Plug-and-play; CC-BY 4.0 ([repo](https://github.com/kevinzakka/mujoco_scanned_objects)) |
| Drawer | Hand-written MJCF, about 50 lines of box geoms plus a `slide` joint | Exact clearances; no convex-hull filling |
| Recording | `mujoco.Renderer` → frames → `mediapy.write_video` | Headless; one renderer per process, because a GL context is single-thread ([MuJoCo Python](https://mujoco.readthedocs.io/en/latest/python.html)) |

Three practical notes for this phase:

- **Drive the arm through actuators.** Feed IK joint targets to the Panda's position actuators as `ctrl`. Never write `qpos` directly, except when debugging.
- **Stop the mug slipping.** Give the finger pads and mug higher friction and `condim` 4–6. This is standard MuJoCo practice, not a sourced number, so tune it.
- **Sequence the drawer work.** Start with the drawer pre-opened, as many papers do. Opening drawers is the hard step: PromptBook reports only 50–71% real drawer-opening success with LLM-written primitives ([PromptBook](https://neurips.cc/virtual/2023/77272)).

### Phase 1, 19 Oct – 1 Nov: reconstruct the twin (≈20–25 h)

Use a **hybrid manual pipeline**. It is the lowest-risk path for a 4–6 week MuJoCo project. The articulated reconstruction systems (DRAWER, iTACO, RialTo, ACDC) target Isaac Sim or OmniGibson, need heavy GPU stacks, and lack MJCF export.

1. **Capture (≈2 h on site).**
   1. Polycam or Scaniverse LiDAR scan of the counter and cabinet, exported in meters.
   2. Photos of the cup from 8–12 angles.
   3. Tape measurements: drawer inner width, depth and height; maximum travel; counter-to-drawer gap; cup diameter, height and mass (kitchen scale).
   4. Optional: a Record3D video of the drawer opening.
2. **Static shell (≈4 h).** Import the scan as a **visual-only** mesh (`contype=conaffinity=0`). Add box collision geoms fitted to the counter and cabinet planes.
3. **Drawer (≈3 h).**
   1. Rebuild the drawer from box primitives at the measured dimensions.
   2. Set the slide axis to the cabinet-front normal, from a plane fit or by measurement.
   3. Set `range` to the measured travel, and add `damping` and `frictionloss`.

   Hand measurement beats every learned method here. Joint limits are the least reliably estimated quantity in the literature, and drawers are almost always prismatic along the front normal.
4. **Cup (≈4 h).**
   1. Get a closed mesh from the scan or from image-to-3D ([TRELLIS.2](https://github.com/microsoft/TRELLIS.2), MIT).
   2. Rescale to the caliper measurements, since image-to-3D output has no scale.
   3. Run [CoACD](https://github.com/SarahWeiii/CoACD) with `real_metric=True` and a threshold of 2–5 mm to keep the cavity open.
   4. Emit MJCF with [obj2mjcf](https://github.com/kevinzakka/obj2mjcf).
   5. Use `shell` or `exact` mesh inertia. The default `legacy` setting overcounts volume for non-convex meshes ([MuJoCo XML reference](https://github.com/google-deepmind/mujoco/blob/main/doc/XMLreference.rst)).
5. **Validate (≈3 h).**
   1. Shake test: no free body moves more than a few mm in 1–2 s.
   2. Table of real-vs-sim clearances with errors in mm.
   3. Optional half-day system ID: match how far the drawer slides and the cup's sliding angle by tuning `frictionloss` and `friction`.

**Stretch only:** [LiteReality-Agent](https://github.com/LiteReality/LiteReality-Agent/) is the one tool found that goes from an iPhone scan to an articulated MuJoCo `scene.xml` end to end, and it has a kitchen example. It is only weeks old and needs OpenAI/Gemini keys, Modal, the codex and claude CLIs, and Blender 5.x. [EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen) offers per-object MJCF conversion with VLM-estimated mass and friction; use its `cu128` build on Blackwell.

### Phase 2, 26 Oct – 8 Nov: variant generator (≈20 h)

Build variants by loading *S0* with `MjSpec.from_file`, editing, and compiling. Specifically:

- Delete or attach bodies for cup cousins and clutter.
- Set the drawer opening through the joint's `qpos`.
- Scale friction and mass on the compiled `MjModel`.
- Save each variant with `spec.to_zip()` plus a JSON file of its parameters ([MuJoCo Python](https://mujoco.readthedocs.io/en/stable/python.html), [mjspec tutorial](https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb)).

Two API cautions. Use `MjVfs` rather than the deprecated assets dict. `MjSpec.copy()` and `first_body()` were not verified; fall back to re-parsing the base XML for each variant.

| Axis | Development range (Sobol-sampled) | Held-out out-of-distribution (OOD) cell |
|---|---|---|
| Cup pose | x-y within a reachable 20×20 cm counter zone (the GaP range ([GaP](https://arxiv.org/html/2607.05369v1))); yaw 0–360°; 3–4 GSO mug cousins that fit the drawer | Positions just outside the zone; an unseen mug |
| Drawer opening | 0–80% of travel | 80–100% of travel; handle offset by ±1 cm |
| Clutter | 0–4 GSO objects on the counter, 0–1 in the drawer, outside a keep-out corridor | 5–6 objects; unseen object identities |

**Check every variant in cost order, rejecting at the first failure.** These steps follow the layered checks in Scenethesis, ACDC and SetItUp:

1. **Analytic fit (microseconds).** The cup's radius plus margin fits within half the drawer's width; the drawer's free space is at least the cup's footprint; the cup is in the IK workspace.
2. **Spawn collision.** After `mj_forward`, there are no new contacts, and the path corridor between cup and drawer is free of clutter.
3. **Settle.** After 0.5–1 s with no robot action, no object has moved more than 1 cm or rotated more than 5°.
4. **Oracle.** The Phase 0 scripted solver succeeds.
5. **Log.** Record every rejection reason. A high rejection rate on one parameter means its range is too wide.

Report coverage in three ways:

1. Parameter-space coverage, as the share of grid cells occupied.
2. The number of distinct cup and clutter assets. Diversity of environments and objects predicts generalization better than the number of demos ([Lin et al.](https://arxiv.org/abs/2410.18647)).
3. Optionally, the DINOv2 nearest-neighbour distance from each test scene to the development set, following ACDC's analysis.

### Phase 3, 2 – 15 Nov: agent loop (≈25 h)

The loop follows ASPIRE, GaP and DynaHarness, cut down to fit a course project:

1. **Prompt.** The LLM gets the task, the skill API documentation, the object names, and one example program. It writes `policy.py`.
2. **Simulate.** The harness runs the program on every development episode in a sandbox, using one process per CPU core with no rendering. It renders an MP4 only for failures.
3. **Feedback.** The agent receives structured results per episode:
   - the stage reached
   - each primitive's inputs and outputs
   - robot–environment contacts
   - the final cup pose relative to the drawer box
   - a failure label
4. **Revise.** The agent edits the program. A revision is accepted only if it does not reduce success on the full development set. This regression gate is borrowed from [DynaHarness](https://arxiv.org/html/2609.40306).
5. **Stop.** The run ends at a fixed budget, for example 15 revisions, which is the same for every condition.

Drive the LLM from a plain Python loop over an API rather than an interactive session, so every token and revision is counted. **The success predicate and the test sets live outside the agent's sandbox, and the agent cannot read them.**

**Make the skill API mid-level so the variation actually matters.** CaP-X found success "degrades as these priors are removed" across 12 models ([CaP-X](https://proceedings.mlr.press/v306/fu26u.html)). That cuts both ways. A monolithic `place_in(drawer)` would make both conditions trivially succeed. Expose these primitives instead, so the agent itself must decide how far to open the drawer, where to grasp, and how to route around clutter:

- `get_pose(name)`, with Gaussian noise of about 5–10 mm to mimic perception
- `get_drawer_state()`
- `get_handle_pose()`
- `move_to(pose)`, with mink collision avoidance
- `grasp()`, `release()`, `pull_along(axis, d)`

### Phase 4, 9 – 23 Nov: evaluation and write-up (≈25 h)

Run all conditions and ablations, using the protocol below. Then freeze and make the demo video (≈4 h) and write the report (≈8 h). Leave **one buffer week inside this window**, since 9 Nov is a public holiday and the 2 Nov lecture covers agentic programming.

## Evaluation protocol: equal budget, pre-registered held-out sets

The comparison holds five things fixed: the LLM, the prompt, the skill API, the revision budget, and the number of development episodes. **Only the development scene distribution changes.** This copies ACDC's Twin / Twin+DR / Cousin design ([ACDC](https://arxiv.org/html/2410.07408)) and ASPIRE's split between debug seeds and disjoint evaluation seeds ([ASPIRE](https://research.nvidia.com/labs/gear/aspire/)).

| Condition | Development set (32 episodes each) | Role |
|---|---|---|
| B0 one-shot | No simulation feedback | Code-as-Policies floor |
| B1 twin-only | *S0* × 32 noise seeds (perception and physics noise only) | Main baseline (ACDC "Twin") |
| B2 twin + naive randomization | *S0* with uniform noise on all parameters, without task-preservation checks | ACDC "Twin+DR" analogue |
| **M variant family** | 32 verified Sobol variants | Proposed method |
| M+memory (optional) | M plus an ASPIRE-style skill library | Stretch |

**Splits, fixed before any agent run and hashed in the log:**

| Split | Contents | Size |
|---|---|---|
| *T_id* | Fresh Sobol seeds drawn from the development ranges | 100 |
| *T_ood* | Four cells: cup, drawer, clutter, all combined | 4 × 100 |
| *T_proxy* | Hidden "reality gap" scenes (below) | 100 |
| Fidelity probes | *T_proxy* with one perturbation type switched on at a time | 5 × 50 |

**Sim-only proxy for unseen real configurations.** *T_proxy* perturbs things that the development family never varies. It follows ACDC's held-out setups at increasing distance:

- drawer and cabinet pose or mesh error of 1–3 cm and 1–5°, mimicking reconstruction error
- drawer friction and damping ×0.5–1.5, and a different handle shape
- a different cup mesh, and mass ×0.7–1.3
- counter height ±2 cm
- larger perception noise

These magnitudes are a team design choice; no published standard exists. Set them from the measured Phase 1 clearance errors. Running the fidelity probes one perturbation at a time answers research question 3, which fidelity dimension matters most, directly in simulation.

**Metrics:**

1. **Primary: success rate.** The cup is inside the drawer box, tilted 20° or less (a team-chosen threshold), and released, and no object has fallen off the counter. A stricter version also requires the drawer to be closed within 1 cm.
2. **Partial progress:** the stage reached, in the spirit of ProgPrompt's goal-condition recall ([ProgPrompt](https://ar5iv.labs.arxiv.org/html/2209.11302)).
3. **Safety:** robot–clutter and robot–cabinet contacts, and objects displaced by more than 2 cm.
4. **Generalization gap:** development success minus held-out success. ACDC's twin went from 100% to 25%.
5. **Cost:** revisions, tokens, and wall-clock time to reach the final program.

**Ablations, run in this order and stopping when time runs out:**

1. Leave one axis out: M without cup variation, without drawer variation, without clutter.
2. Family size *K* = 1, 4, 8, 16, 32, mirroring ACDC's ablation over the number of cousins.
3. Feedback richness: pass/fail only, then stage plus failure labels, then full traces.
4. Regression gate on versus off.
5. API level: mid-level versus a monolithic `place_in()` call, to show the ceiling effect explicitly.

**Trials and statistics.** Physics is deterministic for a fixed program, so the variance that matters comes from the *agent*.

- **Agent runs:** run each condition **≥3 times, 5 if the budget allows**, with different LLM samples.
- **Test size:** score every final program on the same ≥100 configurations per cell, which gives roughly ±6–10 pp CI half-widths.
- **Reporting:** per-run k/n with Wilson 95% CIs, plus a mean over runs from a two-level bootstrap that resamples runs and then configurations.
- **Main test:** M vs B1 is paired on identical configurations, so use McNemar's test or a paired permutation test ([VoLo](https://arxiv.org/html/2606.07723v1) uses the latter).

| Successes | Wilson 95% CI | Successes | Wilson 95% CI |
|---|---|---|---|
| 18/20 | 69.9–97.2% | 90/100 | 82.6–94.5% |
| 15/20 | 53.1–88.8% | 45/50 | 78.6–95.7% |
| 10/20 | 29.9–70.1% | 40/50 | 67.0–88.8% |

**If a real robot becomes available,** run 10–20 trials per condition on 2–3 matched physical configurations. Use the same configurations for every condition, as ACDC (n=20) and GaP (20–30) did. At 20 trials, 18/20 vs 10/20 is separable but 18/20 vs 15/20 is not. Use [SureSim](https://arxiv.org/abs/2510.04354) to combine a few paired real trials with large simulation runs and correct for simulator bias. Use [STEP](https://arxiv.org/abs/2503.10966) for sequential stopping, which needs up to 32% fewer trials. Which robot is available, if any, is still unknown and must be confirmed.

## Related work and the gap this project fills

**Real-to-sim.** The field has four families:

1. Retrieval-based "cousins": [ACDC](https://arxiv.org/html/2410.07408v2), [Articulate-Anything](https://arxiv.org/pdf/2410.13882), [URDFormer](https://github.com/WEIRDLabUW/urdformer).
2. Reconstruction plus articulation from video: [DRAWER](https://arxiv.org/html/2504.15278v2), [iTACO](https://github.com/3dlg-hcvc/video2articulation), [Real2Code](https://arxiv.org/html/2406.08474v2).
3. Gaussian-splat rendering over a physics simulator: [SplatSim](https://github.com/qureshinomaan/SplatSim), [RialTo](https://arxiv.org/html/2403.03949v3).
4. 2026 agentic whole-room builders: LiteReality-Agent.

Almost all of them target Isaac Sim, OmniGibson, SAPIEN or PyBullet. Only LiteReality-Agent, EmbodiedGen (via a converter) and End-to-End Real-to-Sim ([arXiv 2506.04120](https://arxiv.org/pdf/2506.04120), which uses differentiable MuJoCo) emit MuJoCo models. On fidelity, the evidence says functional geometry and diversity matter more than visual exactness. High visual fidelity "does not by itself guarantee reliable manipulation learning" ([arXiv 2609.18293](https://arxiv.org/pdf/2609.18293)). That justifies spending effort on clearances, the joint axis and contact, and not on textures.

**Scene variation.** Variation methods have moved through three stages:

1. Classic domain randomization ([Tobin et al.](https://www.emergentmind.com/papers/1703.06907)).
2. Procedural families ([RoboCasa](https://arxiv.org/html/2406.02523v1): 120 kitchen layout-style combinations and 2,509 objects).
3. "LLM proposes, solver disposes" generators ([PhyScensis](https://arxiv.org/abs/2602.14968), [Scenethesis](https://arxiv.org/html/2505.02836v1)).

The direct evidence for this project's hypothesis is strong. ACDC got 25% real success from a twin, 55% from twin plus randomization, and 95% from twin plus cousins, on a cabinet task. RialTo reached 91% vs 25% for behavior cloning (BC) under pose shifts ([RialTo](https://arxiv.org/html/2403.03949)). One caveat matters for choosing ranges: ACDC also found that training on *all* feasible assets did worse than on selected cousins. More variety helps only while it stays relevant to the target.

**Course reading list.** Each paper contributes a piece of the design:

| Paper | What it contributes here |
|---|---|
| [ProgPrompt](https://arxiv.org/abs/2209.11302), [Code as Policies](https://arxiv.org/abs/2209.07753) | Program-as-policy over a primitive API. ProgPrompt's assertions lifted SR from 0.28 to 0.34 |
| [ApBot](https://arxiv.org/html/2505.20424v2) | Explicit model plus closed-loop update; removing the update cut success from ≈0.93 to ≈0.36 |
| [SetItUp](https://arxiv.org/html/2508.02068v2) | Boolean relation checkers plus samplers, reused as the variant feasibility checks |
| [GaP](https://arxiv.org/html/2607.05369v1), [ASPIRE](https://research.nvidia.com/labs/gear/aspire/) | Rehearse programs over sampled variations, then test on held-out seeds. GaP: 0.97 on mixed variations vs 0.10 for a single-agent CaP-X |
| [ENPIRE](https://research.nvidia.com/labs/gear/enpire/), [VoLo](https://arxiv.org/html/2606.07723v1) | Real-world loop, cost reporting, Wilson CIs and paired tests |

The novelty claim is narrow and defensible. ACDC, RialTo and URDFormer tested twin versus variation for *learned* policies. GaP varies tasks but reports no clear held-out split and no single-scene baseline. No source in the notes compares a coding agent's program developed on a twin against one developed on a variant family, using held-out tests.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| robosuite downgrades numpy and pins mink 0.0.5 in `robot` | Keep `robot` robosuite-free. If RoboCasa kitchens are wanted as assets, create a separate `robocasa` env; its install needs robosuite master and about 10 GB of assets ([robocasa](https://github.com/robocasa/robocasa)) |
| Hollow objects collide as solid hulls | Build the drawer from boxes; decompose the cup with CoACD and visually check the cavity stays open |
| Both conditions hit 100% (ceiling) | Mid-level API, perception noise, OOD cells, and the API-level ablation as proof |
| Agent games the metric | The success check and test sets sit outside the sandbox; the agent never sees them |
| Rejection sampling favours easy variants | Use a separate oracle from the agent; report rejection rates per axis; keep a "hard" stratum in *T_ood*. No paper quantifies this bias |

Three smaller risks:

- **GPU stacks on Blackwell.** Heavy repos pinned to old torch builds (URDFormer, original GroundingDINO) need torch ≥2.7 with cu128 on Blackwell ([PyTorch forum](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)). Keep them off the critical path.
- **Multi-GPU rendering.** Pin each worker process to one GPU with `MUJOCO_EGL_DEVICE_ID`. This setting comes from general practice and was not re-verified.
- **Agent variance.** Run-to-run variance can hide the effect. Use ≥3 runs and paired tests.

## Conclusion

The way to prove variation helps is to make the twin-only agent's overfitting *observable*. That requires a skill API low enough for scene-specific assumptions to creep in, and test sets the agent can never inspect. Reconstruction quality matters less than it first appears. A tape-measured box drawer beats any learned articulation estimator on the clearances that decide success, which frees most of the 8 weeks for the variation and evaluation work where the new result lies.

The fidelity probes add a second, publishable-quality finding at little extra cost. Switching on one reality-gap perturbation at a time shows *which* fidelity dimension a program-writing agent is most sensitive to. Prior work has measured that for learned policies, not for generated code. Lock the held-out sets and success predicate before the first agent run; everything else can slip a week without endangering the headline comparison.
