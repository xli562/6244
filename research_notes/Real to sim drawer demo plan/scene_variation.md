# Task-Preserving Scene Variation for Robot Simulation (Real-to-Sim Kitchen: "Place Cup in Drawer", MuJoCo)

Scope note: Researched 2026-10-16. About 25 tool calls. Primary sources are arXiv abstracts or HTML, MuJoCo docs, the MuJoCo tutorial notebook (downloaded and grepped), and GitHub READMEs. Claims are marked "(not re-verified this session)" when only the paper's existence and identity were confirmed and the details come from background knowledge. The report writer should treat those as lower confidence.

## Q1. What are the leading approaches (DR, procedural generation, LLM/VLM-driven generation, digital cousins, constraint-based layout incl. SetItUp)?

### Takeaway
The field has moved through three stages. Classic visual and physical domain randomization (DR) came first (2017–2019). Procedural or "templated" scene families followed (RoboCasa, ProcTHOR). The current stage (2024–2026) uses LLM/VLM agents that propose layouts and then verify them with a physics or geometry solver (Scenethesis, PhyScensis, SAGE), alongside real-to-sim "digital cousin" pipelines (ACDC). For a single-scene, few-weeks MuJoCo project, the best fit is a hybrid: a parametric or procedural generator around the reconstructed twin, plus ACDC-style "cousin" asset swaps, plus explicit feasibility constraints. LLM scene generation is optional and adds the least value per week of effort.

### Cited Findings

**Classic domain randomization (older, still the baseline)**
- Tobin et al. (2017) trained detectors on simulated images alone, with non-realistic random textures and randomized distractors, camera pose and lighting. They report about 1.5 cm real-world localization accuracy. — [arXiv 1703.06907 summary](https://www.emergentmind.com/papers/1703.06907)
- A 2023 visual sim-to-real manipulation study randomized table, wall, robot and floor textures but kept *object* textures fixed, because object color was task-relevant. It compared procedural textures against about 1,200 realistic ambientCG textures and randomized light position. This is an early example of "task-preserving" visual DR, where task-relevant attributes are not randomized. — [arXiv 2307.15320](https://arxiv.org/pdf/2307.15320)
- A 2026 empirical study (Jin et al., arXiv 2603.22876, v2 June 2026) breaks sim-to-real generalization into four dimensions: multi-level DR, photorealistic rendering, physics-realistic modeling, and RL updates. It used more than 10k real-world trials varying background, lighting, distractors, object types and spatial features. The abstract does not rank the factors. — [arXiv 2603.22876](https://arxiv.org/abs/2603.22876)
- Search-result summary of the same 2026 study: its DR has five components (background textures, distractor objects, camera pose, lighting, table height) and compares per-episode with per-step resampling. — [arXiv 2603.22876 PDF](https://arxiv.org/pdf/2603.22876)
- Physical or dynamics randomization (Peng et al. 2018), automatic domain randomization (ADR; OpenAI 2019) and the GSO asset set are the canonical older references (not re-verified this session). — [Peng et al. arXiv 1710.06537](https://arxiv.org/abs/1710.06537); [OpenAI ADR arXiv 1910.07113](https://arxiv.org/abs/1910.07113)

**Procedural scene generation**
- RoboCasa (2024, built on robosuite and MuJoCo):
  - Scenes: 10 kitchen floor plans × 12 styles = 120 kitchen scenes, plus 100 AI-generated (MidJourney) textures each for walls, floors, counters and cabinet panels. The authors call the textures "a form of domain randomization."
  - Assets: 2,509 Objaverse and Luma.ai assets across 153 categories.
  - Demo collection: each demo was collected in a random kitchen. — [RoboCasa HTML](https://arxiv.org/html/2406.02523v1)
- RoboCasa defines 100 tasks, some composite tasks generated with LLM guidance. It uses text-to-3D for objects and text-to-image for textures. — [RoboCasa abstract](https://arxiv.org/abs/2406.02523)
- Sceniris (arXiv 2512.16896, Dec 2025) is a "fast procedural scene generation framework." Its comparison table characterizes GenSim2's scene composition as "predefined layouts with small-range sampling." — [Sceniris](https://arxiv.org/html/2512.16896)
- ProcTHOR (2022) and Infinigen-Indoors (2024) are the canonical procedural house and indoor generators. Infinigen-Indoors uses a constraint-specification language plus a solver. — [ProcTHOR arXiv 2206.06994](https://arxiv.org/abs/2206.06994); [Infinigen Indoors arXiv 2406.11824](https://arxiv.org/abs/2406.11824) (not re-verified this session)

**LLM/VLM-driven scene and task generation**
- RoboGen (2023) runs a "self-guided propose-generate-learn cycle." An LLM proposes tasks, generates environments by placing objects and assets in spatial configurations, decomposes tasks into subtasks, and chooses RL, motion planning or trajectory optimization to learn them. — [RoboGen arXiv 2311.01455](https://arxiv.org/abs/2311.01455)
- GenSim2 (CoRL 2024) uses coding LLMs with multimodal and reasoning capabilities to generate long-horizon articulated tasks, "up to 100 articulated tasks with 200 objects."
  - Demonstrations come from planning and RL solvers.
  - Its PPT policy (proprioceptive point-cloud transformer) transfers zero-shot sim-to-real.
  - Co-training with generated data improves performance by 20% over limited real data alone. — [GenSim2 arXiv 2410.03645](https://arxiv.org/abs/2410.03645)
- GenSim2 prompts an LLM to propose a task and then assembles the required objects (as characterized by the PRISM paper). — [PRISM arXiv 2607.04880](https://arxiv.org/pdf/2607.04880)
- GenSim v1 (2023, tabletop Ravens/CLIPort) and Gen2Sim (2023) are earlier versions of this line. They are older and largely superseded by GenSim2. — [GenSim arXiv 2310.01361](https://arxiv.org/abs/2310.01361); [Gen2Sim arXiv 2310.18308](https://arxiv.org/abs/2310.18308) (not re-verified this session)
- Holodeck (CVPR 2024) is an LLM-driven generator of indoor 3D scenes using Objaverse assets.
  - In Scenethesis' comparison it has 6.1% object-object and 21% small-object collision rates.
  - It "focuses only on large-object collision avoidance."
  - It supports only "on top of" as a vertical relation.
  - These criticisms come from competitors' papers. — [Scenethesis](https://arxiv.org/html/2505.02836v1); [SAGE arXiv 2602.10116](https://arxiv.org/html/2602.10116v1)
- Scenethesis (NVIDIA, ICLR 2026) is training-free and agentic. An LLM plans a coarse layout, vision guidance refines it, and physics-aware optimization (SDF collision, scale and stability losses) cleans it up. A GPT-4o judge re-plans when scores fall below a threshold. — [Scenethesis](https://arxiv.org/html/2505.02836v1)
- PhyScensis (ICLR 2026, arXiv 2602.14968):
  - Loop: an LLM agent proposes assets with spatial *and physical* predicates (contact, support, balance, containment). A physics-engine-backed solver realizes them, and solver feedback is used to refine the scene.
  - Stability: probabilistic programming enforces it.
  - Target: tabletop, shelf and box-packing manipulation scenes, which is the most relevant LLM approach for the "cup + drawer + clutter" setting. — [PhyScensis](https://arxiv.org/abs/2602.14968); [project page](https://physcensis.github.io)
- SAGE (arXiv 2602.10116, Feb 2026) offers "scalable agentic 3D scene generation for embodied AI." It criticizes Holodeck for producing fewer objects, lower realism and frequent collisions. — [SAGE](https://arxiv.org/html/2602.10116v1)
- A 2026 survey of VLA data engines concludes that "physical grounding, embodiment alignment, and reliability verification lag behind scalability." — [arXiv 2604.23001](https://arxiv.org/html/2604.23001v1)

**Digital cousins (ACDC, CoRL 2024)**
- Definition: a digital cousin is a virtual asset or scene that does not replicate the real scene but preserves its geometric and semantic affordances.
- Pipeline (single RGB image), extraction step: GPT-4(o) captions objects, Grounded-SAM-v2 produces masks, Depth-Anything-v2 produces depth and a point cloud, and DBSCAN removes noise.
- Pipeline, matching step: CLIP text similarity picks the top-k categories, then DINOv2 feature distance ranks candidate assets from BEHAVIOR-1K (more than 10k assets) across orientation snapshots.
- Pipeline, scene step: assets are placed at the point-cloud centroid and rescaled to the object's extents; floor and wall planes are fitted; "on-top" relations come from more than 70% footprint overlap; objects are de-penetrated and x-y collisions resolved. Assembly takes under 30 s. — [ACDC HTML](https://arxiv.org/html/2410.07408)

**SetItUp (arXiv 2508.02068; IJRR journal version of RSS 2024; Xu, Mao, Li, Du, Lozano-Pérez, Kaelbling, Hsu; NUS + MIT)**
- Problem: functional object arrangement (FORM) from under-specified instructions, e.g. "set the table for two." — [arXiv 2508.02068](https://arxiv.org/abs/2508.02068)
- Neuro-symbolic, two stages:
  1. An LLM induces a Python *program* from a structured task specification and a few examples. Running the program outputs a **grounding graph** of abstract spatial relations. A coder and verifier pipeline checks the program.
  2. Per-relation **diffusion models**, trained on synthetic data, are composed at inference to sample object poses that satisfy the graph. — [arXiv 2508.02068](https://arxiv.org/abs/2508.02068)
- Relation library: 48 relations. 24 are unary (e.g. central-column, near-front-edge, against-back-wall), 13 binary (left-of, on-top-of, centered, in-front-of…), 2 ternary (symmetry about an axis object), and 9 variable-arity (regular-grid, contiguously-aligned, height-sorted…).
- Each relation has two parts: a rule-based Boolean classifier h_R, used for checking, and a diffusion generator f_R, used for sampling. — [SetItUp HTML](https://arxiv.org/html/2508.02068v2)
- Composition: normalized per-relation scores are summed (an energy-based product of experts, after Du et al. 2023). Sampling uses ULA-style Langevin MCMC with M = 20 updates per noise level. — [SetItUp HTML](https://arxiv.org/html/2508.02068v2)
- Results (physical feasibility % / functionality %), mean over 15 test scenes:

  | Task family | SetItUp | Best baseline |
  |---|---|---|
  | Dining | 94.0 / 89.1 | Direct-LLM-with-spec 82.1 / 67.7 |
  | Bookshelf | 98.1 / 94.4 | LLM(Program)-CompDMs 90.0 / 93.5 |
  | Bedroom | 93.8 / 91.5 | LLM(Program)-CompDMs 45.6 / 78.2 |

  An end-to-end diffusion baseline scored only 22.7 / 21.8 on dining. — [SetItUp HTML](https://arxiv.org/html/2508.02068v2)
- Robot experiments listed in the paper (numbers not extracted):
  - Reachability constraints on a dual-arm platform.
  - A 30+ object buffet scene that reuses the induced dining program unmodified. — [SetItUp HTML](https://arxiv.org/html/2508.02068v2)
- Limitations:
  - The layout is 2D.
  - The 48-relation library is fixed.
  - It needs an expert-written structured task specification.
  - Wrong graphs misdirect diffusion grounding.
  - Physical feasibility is *measured* (2D collision-free and inside-container), not guaranteed by an explicit collision term. — [SetItUp HTML](https://arxiv.org/html/2508.02068v2)

### Inferences
- SetItUp is a layout *generator* for functional arrangements, not a sim-variation tool per se. Its most transferable idea for this project is the pattern of **symbolic relation graph + Boolean checkers + samplers**. Write task-preservation rules as Boolean predicates, for example:
  - `cup_inside_drawer_reachable`
  - `clear(drawer_front, 0.10 m)`
  - `not-blocking(clutter_i, cup → drawer path)`
- Use those predicates both to sample variants (by rejection or optimization) and to verify them. Training diffusion models per relation is overkill for a few-week project; rejection sampling against h_R-style checkers achieves the same effect at this small scale.
- PhyScensis and Scenethesis show that the 2025–2026 consensus is "LLM proposes, physics solver/optimizer disposes." Pure LLM layouts (Holodeck-era) have nontrivial collision and instability rates.
- For a single reconstructed kitchen, ACDC's "cousin" idea maps directly onto the project. Swap the cup for other mugs and cups of similar size from GSO or YCB, and the drawer for similar drawers with different handle, depth and friction. This keeps the semantic affordance (graspable cup, openable drawer) while breaking overfitting to one exact mesh.

### Gaps
- PhyScensis' quantitative numbers and its exact solver (which physics engine; probabilistic-programming details) were not extracted.
- Infinigen-Indoors' constraint language and ProcTHOR statistics were not re-verified this session.
- "ArchitectGPT-like" systems: no specific robotics paper by that name was found. It appears to be an interior-design product rather than a research method.
- GenSim2's solvability verification details (kPAM-based planner and rejection) are not confirmed from the abstract alone.

## Q2. How do methods guarantee variants remain feasible and solvable?

### Takeaway
No method offers formal guarantees. They layer cheap geometric checks, then physics settling and stability checks, then **oracle verification**: a scripted policy, planner or MimicGen replay must succeed in the variant, and the variant is discarded otherwise. Rejection sampling with a scripted solver is the most common and most implementable "task-preserving" guarantee.

### Cited Findings
- Scenethesis uses three differentiable physics-style losses on SDFs:
  - **Collision loss:** 400 surface points sampled per object; points with negative SDF are pushed out toward the centroid.
  - **Scale loss:** objects are shrunk if collisions form multiple clusters.
  - **Stability loss:** bottom points are pulled to zero SDF on the support surface.

  A GPT-4o judge then scores category, orientation and spatial coherence and triggers re-planning below a threshold. — [Scenethesis](https://arxiv.org/html/2505.02836v1)
- Scenethesis ablation: the raw LLM layout had 22.7% collision and 87.3% instability. With all constraints these fell to 0.8% and 3.2%. **Unconstrained generated layouts are mostly infeasible.** — [Scenethesis](https://arxiv.org/html/2505.02836v1)
- Scenethesis measures instability by running a physics simulation and checking displacement (Inst-O/Inst-S, following Atlas3D). It also reports "Reach" and "Walk" interactivity metrics. — [Scenethesis](https://arxiv.org/html/2505.02836v1)
- ACDC de-penetrates objects and resolves x-y collisions so the scene is physically stable. Its demonstrations come from scripted Open, Close, Pick and Place skills with **rejection sampling, keeping only successful episodes**: about 10k demos per task, 2k for the longer bowl-put-away task. — [ACDC HTML](https://arxiv.org/html/2410.07408)
- RoboCasa and MimicGen split each human demo into object-centric segments, transform them to new object poses, stitch them together, and **keep only successful rollouts**. RoboCasa notes that many generated trajectories still contain jerky motion and collisions, and that filtering them via sim-state checks is future work. — [RoboCasa HTML](https://arxiv.org/html/2406.02523v1)
- PhyScensis realizes LLM-proposed spatial and physical predicates with a physics-engine solver and uses probabilistic programming for stability. Solver feedback loops back to the agent. — [PhyScensis](https://arxiv.org/abs/2602.14968)
- SetItUp checks relations with rule-based Boolean classifiers and uses a verifier in the program-induction stage. It *measures* feasibility as the fraction of objects that are collision-free (2D detector) and inside their container. — [SetItUp HTML](https://arxiv.org/html/2508.02068v2)
- SetItUp's robot section adds reachability constraints for a dual-arm platform, so reachability can be added as a relation. — [SetItUp HTML](https://arxiv.org/html/2508.02068v2)
- A 2026 Electronics Letters paper runs LLM-generated tool-use robot code in a physics sim. It detects collisions and misalignments and corrects parameters while keeping the task structure, which amounts to simulation-in-the-loop verification of actions. — [Kim 2026](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ell2.70575)
- RoboGen chooses RL, motion planning or trajectory optimization per subtask. Solvability is then implicitly tested by whether the learner succeeds. — [RoboGen](https://arxiv.org/abs/2311.01455)

### Inferences (recommended verification pipeline for cup-in-drawer)
Run these checks in order of cost. Reject the variant at the first failure.
1. **Parametric pre-checks (analytic, microseconds).**
   - Cup fits: the cup's bounding radius plus margin is at most the drawer inner width/2, and the cup height is at most the drawer inner height. This is the same idea as SetItUp's Boolean h_R.
   - Drawer opening `q_drawer ≥ q_min`, so the opening exceeds the cup diameter plus gripper clearance (e.g. ≥ 2 cm each side).
   - Cup and target lie inside a precomputed reachability box or inverse-kinematics workspace of the arm.
2. **Collision check at spawn.** Call `mj_forward`, then require `data.ncon == 0` among the new clutter bodies, or a minimum `mj_geomDistance` ≥ clearance. The cup-to-drawer path corridor (a swept box) must be clutter-free.
3. **Physics settling.** Simulate 0.5–1 s with no robot action and reject if any object moves more than about 1 cm or rotates more than about 5°. This mirrors Scenethesis' Inst-O metric.
4. **Oracle solvability.** Run a scripted or IK-based pick-place (or a motion planner) and keep the variant only if it succeeds. This is the ACDC and MimicGen rejection-sampling approach, and it also yields demos for imitation learning for free.
5. Log the rejection reasons. A high rejection rate on one parameter means its range is too wide.

### Gaps
- None of the sources gives a formal completeness guarantee. Rejection sampling can silently bias the variant distribution toward "easy" configurations for the scripted oracle. No paper quantifying this bias was found.

## Q3. How do they measure variant diversity and coverage?

### Takeaway
There is no standard metric. Papers use one of three proxies:
- counts of distinct environments and objects (scaling-law studies);
- embedding distance between sim and real scenes (ACDC's DINOv2 distance);
- LLM or human preference judgments of "object diversity" (Scenethesis).

The project should define its own parameter-space coverage plus an embedding-space distance to the real test configurations.

### Cited Findings
- Lin et al. (2024) measure diversity by the **number of distinct training environments and objects**. Generalization follows a roughly power law in these counts. — [arXiv 2410.18647](https://arxiv.org/abs/2410.18647)
- ACDC uses **DINOv2 feature distance** both to select cousins and as an axis of analysis: digital-twin policy performance drops as the DINOv2 distance of the test setup grows. DINO-based selection generally outperformed CLIP-based selection. — [ACDC HTML](https://arxiv.org/html/2410.07408)
- ACDC also found that training on *all* feasible assets did worse than training on the selected cousins. **More diverse is not monotonically better; relevance to the target distribution matters.** — [ACDC HTML](https://arxiv.org/html/2410.07408)
- Scenethesis assesses diversity and realism through GPT-4o and a 69-person human preference study over 22 scenes. It was preferred over Holodeck on "object diversity" 85% (GPT-4o) and 80% (human) of the time. It also reports CLIP, BLIP and VQA text-alignment scores. — [Scenethesis](https://arxiv.org/html/2505.02836v1)
- RoboCasa's diversity comes by construction: 120 layout-style combinations, 400 AI textures and 2,509 objects. It evaluates on five fixed scenes, two of which have styles unseen in training, using unseen object instances. — [RoboCasa HTML](https://arxiv.org/html/2406.02523v1)

### Inferences
Practical metrics for the project:
1. **Parameter-space coverage.** Normalize each variation parameter to [0,1], e.g. (cup x, cup y, cup yaw, drawer opening, number of clutter objects, clutter positions, friction, mass). Report a coverage measure such as the fraction of occupied grid cells or a star-discrepancy score. Sobol or Latin-hypercube sampling gives better coverage than i.i.d. uniform sampling for the same N.
2. **Perceptual distance to real.** Embed rendered variant images and photos of the *held-out real configurations* with DINOv2. Report the nearest-neighbor distance from each real test configuration to the variant family versus to the original twin. Smaller distances indicate that the family "covers" reality, following ACDC's analysis.
3. **Count of distinct cup and clutter assets.** Following Lin et al., report how many distinct objects appear. This is likely the most predictive knob.

### Gaps
- No source validated a single diversity metric as predictive of real-world success. The DINOv2-distance correlation in ACDC is the closest evidence, and its numeric correlation was not extracted.

## Q4. Evidence that variation helps real-world generalization

### Takeaway
The evidence is strong and consistent. **Diversity of environments and objects beats demo count** (Lin et al. 2024), and **digital cousins beat a single digital twin by a wide margin in zero-shot real transfer** (90% vs 25%, ACDC). This directly supports the project's success criterion.

### Cited Findings
- Lin et al., "Data Scaling Laws in Imitation Learning for Robotic Manipulation" (arXiv 2410.18647, Oct 2024):
  - Generalization follows a roughly power law in the number of training environments and objects.
  - "The diversity of environments and objects is far more important than the absolute number of demonstrations."
  - Beyond a certain number of demos per environment or object, more demos add little.
  - Four collectors working one afternoon gathered enough data for about 90% success in novel environments with unseen objects on two tasks.
  - The study involved more than 40k demos and more than 15k real rollouts. — [arXiv 2410.18647](https://arxiv.org/abs/2410.18647)
- ACDC real-world zero-shot door opening on an IKEA cabinet (sim n = 50, real n = 20):

  | Training scenes | Sim success | Real success |
  |---|---|---|
  | Twin | 100% | 25% |
  | Twin + DR | 70% | 55% |
  | Twin + Cousin | 92% | 95% |
  | Cousin | 94% | 90% |

  — [ACDC HTML](https://arxiv.org/html/2410.07408); [abstract: "90% vs. 25%"](https://arxiv.org/abs/2410.07408)
- In ACDC simulation, cousin policies match twin policies in-distribution and degrade less on unseen setups. Its tasks include **Drawer Opening** and **Putting Away Bowl**, which are close analogues of cup-in-drawer. — [ACDC HTML](https://arxiv.org/html/2410.07408)
- RoboCasa:
  - Multi-task BC on atomic tasks scales with generated data volume: Human-50 28.8%, Gen-100 26.3%, Gen-300 35.0%, Gen-3000 47.6%.
  - Real-world co-training lifted success from 13.6% to 24.4% on seen objects and from 2.6% to 9.3% on unseen objects. — [RoboCasa HTML](https://arxiv.org/html/2406.02523v1)
- GenSim2: co-training with generated sim data gives +20% over limited real data alone. — [GenSim2](https://arxiv.org/abs/2410.03645)

### Inferences
- **Twin + DR helps but less than Twin + Cousins.** In ACDC, 55% versus 95% real success, and DR also lowered sim success to 70%. Randomizing *structure and assets* (positions, drawer state, object instances, clutter) is likely more valuable than randomizing appearance and physics alone. The project should prioritize geometric and asset variation, then add moderate visual and physical DR.
- Expected outcome for the project's success criterion: a policy trained only in the original scene behaves like ACDC's "Twin" (high in-scene, poor off-scene). A variant-family policy should resemble "Cousin." Evaluation needs held-out real configurations: new cup positions, a different drawer opening, unseen clutter.
- Caution: ACDC's and Lin's evaluation scale (many environments) is far larger than a course project's, so effect sizes will be smaller and noisier. Use at least about 10–20 real trials per condition, matching ACDC's n = 20.

### Gaps
- No source specifically quantifies how *many* variants are needed for a single-scene family. Lin et al.'s power-law exponent was not extracted from the abstract.
- The factor-level ranking in the 2026 DR study (arXiv 2603.22876) was not available from the abstract.

## Q5. Practical MuJoCo implementation (MjSpec, randomization, asset libraries)

### Takeaway
MuJoCo ≥ 3.2 `MjSpec` lets you load the reconstructed scene MJCF, procedurally add, delete and attach bodies (clutter, alternative cups, Menagerie arms), compile, and even `recompile` while preserving state. Joint positions such as the drawer opening and physics parameters such as friction and mass can be randomized on the compiled `MjModel`/`MjData` without rebuilding. `mujoco_scanned_objects` provides 1,030 ready-to-use GSO MJCF models for clutter and cup cousins.

### Cited Findings
**MjSpec history and status**
- MuJoCo 3.2.0 (July 2024) introduced `mjSpec` for procedural model creation and editing, initially flagged as "in testing" with possible breaking changes.
- 3.2.1 added an assets-dict argument in Python.
- 3.3.x docs call the API "feature complete and recommended for general use." — [MuJoCo changelog 3.2.2](https://mujoco.readthedocs.io/en/3.2.2/changelog.html); [Model Editing docs](https://mujoco.readthedocs.io/en/3.3.3/programming/modeledit.html)
- The current Python docs mark the old assets dict as **deprecated** ("will be removed in the next release") in favor of `mujoco.MjVfs`. Use `vfs`, not `assets`, on current versions. — [MuJoCo Python docs](https://mujoco.readthedocs.io/en/stable/python.html)

**Model editing workflow**
- Workflow: parse or create a spec, edit it, compile it. The spec stays editable after compilation.
- `mj_recompile` updates the model and data while preserving simulation state, which allows adding or removing bodies mid-simulation.
- Attachment is a shallow move by default; deep copy is opt-in. — [Model Editing docs](https://mujoco.readthedocs.io/en/stable/programming/modeledit.html)

**Python API**
- Construction: `mujoco.MjSpec()`, `MjSpec.from_string(xml)`, `MjSpec.from_file(path)`.
- Building: `spec.worldbody.add_body(pos=..., quat=...)`, `body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=..., rgba=...)`, `model = spec.compile()`.
- Named access: `spec.geom('name')`, `spec.body()`, `spec.joint()`, `spec.site()`; these return None if the element is absent.
- Deletion: `spec.delete(spec.geom('my_geom'))` also removes children and referencing elements.
- Attachment: `frame.attach_body(child_body, prefix, suffix)`, `body.attach_frame(...)`, `parent_spec.attach(child_spec, site=... | frame=...)`, with `spec.copy_during_attach = True` to copy.
- Export: `spec.to_xml()`, `spec.encode('model.xml')`, `spec.to_zip()` / `MjSpec.from_zip()`. — [MuJoCo Python docs](https://mujoco.readthedocs.io/en/stable/python.html)

**Official tutorial notebook (`python/mjspec.ipynb`, grepped this session)**
- Joints and free bodies: `body.add_joint(name="swing", type=mj.mjtJoint.mjJNT_HINGE, axis=[1,-1,0], pos=[...])` and `ball.add_freejoint()`.
- Spawning loop: free bodies are spawned at random positions in a loop (`pos=[x, y, 4+np.random.uniform(0,10)]`).
- Procedural meshes: `spec.add_mesh(name=..., uservert=...)`.
- Assets and defaults: `spec.add_material(...)` and `spec.add_texture(...)`. Defaults are set via `spec.default.geom.type = ...`.
- Attaching robots: `attach_body(franka_arm, 'franka', '')` attaches a Franka loaded with `MjSpec.from_file`.
- State-preserving recompile: `spec.recompile(model, data)`, which returns a new `mjData` holding the mapped state.
- Removal: `spec.delete(body)`. — [mjspec.ipynb](https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb)

**Asset libraries**
- `kevinzakka/mujoco_scanned_objects` contains MJCF models for **all 1,030** Google Scanned Objects.
  - Generation: auto-generated with `obj2mjcf`.
  - Collision geometry: V-HACD convex decomposition into 32 collision submeshes per object.
  - Files per object: `model.xml`, `model.obj` (visual), `model_collision_*.obj`, `texture.png`.
  - License: CC-BY 4.0 for assets, MIT for the XML. — [mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects)
- RoboCasa (MuJoCo/robosuite) ships 2,509 filtered Objaverse and Luma assets across 153 categories plus kitchen fixtures (drawers, cabinets). It is a ready source of MuJoCo-compatible kitchen objects and articulated drawers. — [RoboCasa HTML](https://arxiv.org/html/2406.02523v1)
- ACDC matched against BEHAVIOR-1K (more than 10k assets), but that library targets OmniGibson/Isaac rather than MuJoCo. — [ACDC HTML](https://arxiv.org/html/2410.07408)

### Inferences (implementation sketch; API names verified above unless noted)

```python
import mujoco as mj, numpy as np
rng = np.random.default_rng(seed)

base = mj.MjSpec.from_file("kitchen_twin.xml")   # reconstructed scene with drawer slide joint "drawer_slide"

def make_variant(base, cfg):
    spec = base.copy() if hasattr(base, "copy") else mj.MjSpec.from_file("kitchen_twin.xml")  # copy API unverified; reparsing is safe
    # 1) cup cousin: delete original cup body, attach a GSO mug of similar size
    spec.delete(spec.body("cup"))
    cup_spec = mj.MjSpec.from_file(f"mujoco_scanned_objects/models/{cfg.cup_asset}/model.xml")
    frame = spec.worldbody.add_frame(pos=cfg.cup_pos, quat=cfg.cup_quat)
    cup = frame.attach_body(cup_spec.worldbody.first_body(), "cup-", "")   # first_body() as used in notebook (.first_geom() seen); verify
    cup.add_freejoint()
    # 2) clutter
    for i, (asset, pose) in enumerate(cfg.clutter):
        c = mj.MjSpec.from_file(f"mujoco_scanned_objects/models/{asset}/model.xml")
        f = spec.worldbody.add_frame(pos=pose[:3], quat=pose[3:])
        b = f.attach_body(c.worldbody.first_body(), f"clutter{i}-", "")
        b.add_freejoint()
    model = spec.compile()
    data = mj.MjData(model)
    # 3) drawer opening: set joint qpos directly
    data.qpos[model.joint("drawer_slide").qposadr[0]] = cfg.drawer_open
    # 4) physics DR on compiled model (no recompile needed)
    gid = model.geom("cup-...").id; model.geom_friction[gid, 0] *= cfg.mu_scale
    mj.mj_forward(model, data)
    return model, data
```

- **Task-preserving parameter sampling** (pairs with the Q2 checks):
  - Sample `drawer_open ∈ [cup_diam + 2·clearance, max_travel]`.
  - Sample the cup pose on the counter region within the arm's reachable workspace (precompute with IK on a grid).
  - Sample cup cousins only from assets whose bounding box fits the drawer interior with margin.
  - Sample clutter count ∈ {0..N} with positions outside a "keep-out" corridor between cup, gripper approach and drawer mouth.
- **Physics DR ranges.** Moderate ranges are a common starting point: friction ×[0.7, 1.3], mass ×[0.8, 1.2]. These are not from a cited source and need tuning. After changing `body_mass`, keep inertia consistent (scale `body_inertia` too). This is standard MuJoCo practice, not verified this session.
- **Visual DR.** Randomize textures and materials of walls, counters and the drawer front with `spec.add_texture` / `spec.add_material` or by editing `model.mat_rgba`. Randomize lights and camera pose slightly. Keep task-relevant appearance (cup color/shape cues) within plausible ranges, following the 2307.15320 practice.
- **Batching.** Generate N specs offline and save each as MJCF/MJZ (`spec.to_zip`) together with a JSON of its parameters, for reproducibility and coverage analysis. Use `spec.recompile(model, data)` only when changing topology mid-episode.
- **Evaluation design matching the success criterion.**
  - Baseline: a policy trained or tuned in the twin only.
  - Treatment: the same budget spread across K variants.
  - Test: held-out real configurations.
  - Report the real success rate and the DINOv2 nearest-neighbor distance of each real test configuration to the training set, mirroring ACDC.

### Gaps
- The exact Python signatures for `MjSpec.copy()` and `body.first_body()` were not verified this session; only `first_geom()` was seen in the notebook. Check against the installed MuJoCo version. Fallback: reparse the base XML per variant.
- Whether `mjs_attach` of GSO models with 32 collision meshes each scales well to more than 10 clutter objects (compile time, contact count) was not measured.
- No source was found comparing per-episode versus per-step DR resampling for MuJoCo manipulation. The 2026 study (2603.22876) studies this, but its results were not extracted.
- MuJoCo Menagerie contents (e.g. the Franka FR3 and Panda models used for the arm) were not re-verified this session. The notebook's Franka attach example confirms the attach pattern works.
