# Real-to-Sim Scene Reconstruction for Articulated, Physically Plausible Manipulation Scenes (state as of Oct 2026)

Scope: methods and open-source tools for turning a captured real kitchen scene into an interactive MuJoCo 3.16 scene for "place a cup in a drawer". The drawer needs a working joint, the cup needs real contact, and placement clearances need to be preserved. The team runs headless Linux on 2x RTX PRO 6000 Blackwell and has about 4–6 weeks.

GitHub metadata (stars, license, last push) was queried from the GitHub REST API on 2026-10-16 (`api.github.com/repos/<owner>/<repo>`). Stars, licenses and push dates below come from that query unless another source is cited.

## Leading real-to-sim / digital-twin / digital-cousin methods (inputs, outputs, code, simulator)

### Takeaway
The field has split into four families:
1. **Retrieval-based "cousins/twins"**: ACDC, Articulate-Anything, URDFormer. These are robust and give clean URDFs, but the geometry is only approximate.
2. **Reconstruction-plus-articulation from video or scans**: DRAWER, iTACO/Video2Articulation, Ditto, PARIS, Real2Code, ArtiTwinSplat. Geometry is truer, but the code is heavy and research-grade.
3. **Gaussian-splat rendering layered on physics meshes**: SplatSim, RialTo, End-to-End Real-to-Sim, GSWorld, Phys2Real.
4. **New 2026 agentic whole-room pipelines**: LiteReality-Agent, which exports MJCF directly.

Almost every system targets Isaac Sim, OmniGibson, SAPIEN or PyBullet. The exceptions that emit MuJoCo/MJCF natively are LiteReality-Agent (Sep 2026), EmbodiedGen (via a converter), and the End-to-End Real-to-Sim paper (MuJoCo differentiable physics). For everything else, plan on a URDF→MJCF step.

### Cited Findings

**Retrieval / VLM-driven (single image or few images → URDF)**
- **ACDC / Digital Cousins (CoRL 2024).**
  - Input: one RGB image plus camera intrinsics.
  - Pipeline: GPT-4/GPT-4o writes captions; Grounded-SAM-v2 masks the objects; Depth-Anything-v2 gives depth, chosen over depth cameras because it handles reflective surfaces better. Assets are matched to BEHAVIOR-1K via CLIP and then DINOv2. Articulated objects are matched only to articulated assets. Assets are placed at the point-cloud centroid, scaled to its extents, and de-penetrated. — [ACDC arXiv 2410.07408](https://arxiv.org/html/2410.07408v2)
  - Result on IKEA cabinet door opening, sim-to-real success: Twin 25%, Twin+DR 55%, Twin+Cousin 95%, Cousin 90%. — [ACDC arXiv](https://arxiv.org/html/2410.07408v2)
  - Code: `cremebrule/digital-cousins` (Apache-2.0, ~254 stars, last push 2025-03). Requires OmniGibson and an "NVIDIA RTX-enabled GPU (recommended 24+ GB VRAM)". — [digital-cousins README](https://github.com/cremebrule/digital-cousins)
  - Stated limitation: performance is bounded by the diversity of the asset dataset. — [ACDC arXiv](https://arxiv.org/html/2410.07408v2)
- **URDFormer (RSS 2024).** "Given an image, whether sourced from the internet or captured with a phone, URDFormer predicts its corresponding interactive 'digital twin' in the URDF format."
  - Uses GroundingDINO for boxes. Only tested with Python 3.9 and torch 1.12.1+cu113.
  - Released checkpoint is "trained on 5 categories".
  - Code: `WEIRDLabUW/urdformer`, MIT, ~205 stars, last push 2024-08. — [URDFormer README](https://github.com/WEIRDLabUW/urdformer); [project](https://urdformer.github.io/)
- **Articulate-Anything (ICLR 2025).**
  - Input: text, image or video.
  - Method: a VLM actor-critic (Gemini, GPT or Claude via API key) writes Python that compiles to URDF. Meshes are retrieved from PartNet-Mobility, not generated.
  - Simulators: PyBullet and SAPIEN.
  - Code: `vlongle/articulate-anything`, ~223 stars, last push 2025-07. The README badge says MIT, but the GitHub API returned no license. — [repo](https://github.com/vlongle/articulate-anything); [arXiv 2410.13882](https://arxiv.org/pdf/2410.13882)
  - The paper uses URDFormer and Real2Code as baselines. — [Articulate-Anything arXiv](https://arxiv.org/pdf/2410.13882)
- **Real2Code (ICLR 2025).**
  - Parts are represented as oriented bounding boxes; a fine-tuned LLM (CodeLlama) predicts joints as code; handles up to 10 articulated parts. — [Real2Code arXiv](https://arxiv.org/html/2406.08474v2)
  - Real-world input is multi-view, pose-free RGB reconstructed with DUSt3R. A fine-tuned SAM handles part segmentation and a PointNet model completes shapes. Training data is PartNet-Mobility.
  - Code: `MandiZhao/real2code`, no license in the API, last push 2025-03. — [Real2Code README](https://github.com/MandiZhao/real2code)

**Reconstruction plus articulation from video, scans or interaction**
- **DRAWER (CVPR 2025).** Input is video of a static indoor scene. An articulation module "identifies articulation types and hinge positions, reconstructs simulatable shapes and appearances and integrates them into the scene."
  - Real-to-sim-to-real demo: the reconstruction is imported into Isaac Sim and transferred to a Franka on a mobile base.
  - Code: `xiahongchi/DRAWER`, MIT, ~174 stars, last push 2025-04. — [DRAWER arXiv 2504.15278](https://arxiv.org/html/2504.15278v2); [CVPR page](https://openaccess.thecvf.com/content/CVPR2025/html/Xia_DRAWER_Digital_Reconstruction_and_Articulation_With_Environment_Realism_CVPR_2025_paper.html)
  - No MuJoCo export was found in the sources checked. — [search summary of DRAWER sources](https://arxiv.org/pdf/2504.15278)
- **iTACO / Video2Articulation (3DV 2026).** "Interactable Digital Twins of Articulated Objects from Casually Captured RGBD Videos."
  - Real capture: iPhone 12 Pro with Record3D for the interaction video and Polycam for surface scans. The 192×256 depth is upsampled with PromptDA.
  - Pipeline: MonST3R produces a moving map; AutoSeg-SAM2 does part segmentation; Grounded-SAM-2 masks the hands; joint parameters are refined with a chamfer loss, run once per joint type.
  - Output: meshes of the whole object, the moving part and the static part. URDF output is not documented, though `yourdfpy` is a dependency.
  - NKSR meshing needs about **48 GB of GPU memory**.
  - Code: `3dlg-hcvc/video2articulation`, ~29 stars, last push 2025-11, no license stated. — [repo](https://github.com/3dlg-hcvc/video2articulation)
- **Ditto (CVPR 2022)**: "Building Digital Twins of Articulated Objects from Interaction". `UT-Austin-RPL/Ditto`, MIT, ~129 stars, last push 2024-12. — [repo](https://github.com/UT-Austin-RPL/Ditto)
- **PARIS (ICCV 2023)**: "Part-level Reconstruction and Motion Analysis". `3dlg-hcvc/paris`, MIT, ~89 stars, last push 2025-02. — [repo](https://github.com/3dlg-hcvc/paris)
- **ArtiTwinSplat (ICRA 2026 workshop, arXiv 2606.24628).**
  - Input: RGB-D videos, with no CAD or annotations.
  - Method: unsupervised recovery of "part structure and joint kinematics from observed motion alone"; output is a 3DGS twin.
  - The abstract names no code release, export format or simulator. — [arXiv 2606.24628](https://arxiv.org/abs/2606.24628)
  - A search-engine summary claimed it covers hinged and sliding joints and exports URDF. That is not confirmed by the abstract page. — [search result](https://arxiv.org/pdf/2606.24628)
- **URDF Synthesis from RGB-D Sequences via Differentiable Joint Inference (arXiv 2606.18861)**, **Kinematify (arXiv 2511.01294**, articulated object generation exporting URDF), and **Articulated 3D Scene Graphs for Open-World Mobile Manipulation (arXiv 2602.16356)** surfaced as 2026 work. Code status was not verified. — [2606.18861](https://arxiv.org/pdf/2606.18861); [Kinematify](https://arxiv.org/pdf/2511.01294); [2602.16356](https://arxiv.org/pdf/2602.16356)
- **Articulate-AnyMesh**: `UMass-Embodied-AGI/Articulate-Anymesh`, ~141 stars, last push 2025-05, no license in the API. — GitHub search API (repo: https://github.com/UMass-Embodied-AGI/Articulate-Anymesh)

**Gaussian-splat or photoreal rendering on top of a physics simulator**
- **SplatSim (ICRA 2025).** "Zero-Shot Sim2Real Transfer of RGB Manipulation Policies Using Gaussian Splatting."
  - Physics runs in PyBullet with a UR robot. Gaussian splats are built from COLMAP, and the PyBullet state drives the splat rendering.
  - Code: `qureshinomaan/SplatSim`, ~161 stars, no license in the API, last push 2025-09. — [SplatSim README](https://github.com/qureshinomaan/SplatSim)
- **RialTo (RSS 2024).** Builds digital twins "on the fly from small amounts of real-world data" and uses inverse distillation plus RL fine-tuning. Reports ">67%" robustness gains. — [RialTo arXiv 2403.03949](https://arxiv.org/html/2403.03949v3)
  - Code: `real-to-sim-to-real/RialToPolicyLearning` (~222 stars, last push 2024-09). Targets **Isaac Sim 2022.2.1 + Orbit** with USD scenes and includes drawer/cabinet tasks (`wooden_drawer_bigger`, `cabinet`). — [RialTo README](https://github.com/real-to-sim-to-real/RialToPolicyLearning)
- **End-to-End Real-to-Sim from Imperfect Robot Data (arXiv 2506.04120).** Combines 3DGS with explicit object meshes and jointly refines geometry, appearance, robot poses and physical parameters, using differentiable rendering plus **differentiable physics in MuJoCo**. Shown on ALOHA 2. — [arXiv 2506.04120](https://arxiv.org/pdf/2506.04120)
- **GSWorld (arXiv 2510.20813)**: a closed-loop photoreal simulation suite for manipulation built on Gaussian splats. — [arXiv 2510.20813](https://arxiv.org/pdf/2510.20813)
- **TwinAligner (arXiv 2512.19390)** has two modules: visual alignment (SDF reconstruction plus editable 3DGS) and dynamic alignment (identifying rigid physics from robot–object interaction). — [TwinAligner](https://arxiv.org/html/2512.19390v1)
- **PhysTwin (ICCV 2025)** covers deformables, so it is not directly relevant. MIT, ~469 stars, active through 2026-08. — [repo](https://github.com/Jianghanxiao/PhysTwin)

**Generative and agentic scene builders (2025–2026)**
- **EmbodiedGen (Horizon Robotics, v2.1.0; V2 paper arXiv 2607.07459).**
  - Inputs: image, text, or mesh plus prompt. Outputs "URDF + mesh(.obj/.glb) + 3DGS(.ply)".
  - Conversion: `MeshtoMJCFConverter` for MuJoCo and Genesis; `MeshtoUSDConverter` for Isaac Sim.
  - Physics: "VLM-inferred physical properties (scale, mass, friction)" plus a convex collision proxy. Articulated generation goes through DIPO (NeurIPS 2025).
  - Hardware: the `cu128` build is recommended for RTX 50-series/Blackwell.
  - Apache-2.0, ~704 stars, active through 2026-08. — [EmbodiedGen README](https://github.com/HorizonRobotics/EmbodiedGen)
- **LiteReality-Agent (arXiv 2610.01863, released 2026-08/10).**
  - Input: a scan from the free LiteReality Scanner iPhone app (RGB, depth and a RoomPlan `room.usdz`).
  - Pipeline: an agentic loop edits a seed room until it matches the capture; a layout stage removes collisions; then it builds a MuJoCo `scene.xml`. Objects are rigid bodies "with mass, inertia, friction, colliders, and joints".
  - Export: the whole room to URDF and MJCF (2026-09-12). The repo includes a kitchen example.
  - Requirements: Apache-2.0. Tested on Linux with a ≥24 GB GPU. TRELLIS and GroundingDINO run on Modal by default. It needs an OpenAI (or Gemini) key, a logged-in `codex` CLI and a `claude` CLI, plus Blender 5.x.
  - Articulated assets "follow Articraft"; joint-parameter details are not documented. — [LiteReality-Agent repo](https://github.com/LiteReality/LiteReality-Agent/)
- **Agentic Real2Sim (2607.19190)** turns recorded robot–object interactions into MuJoCo episode twins using VLM agents. This was seen only via alphaxiv and is unverified. — [alphaxiv](https://www.alphaxiv.org/abs/2607.19190)
- **MotionAnymesh (arXiv 2603.12936)**: "Physics-Grounded Articulation for Simulation-Ready Digital Twins". Details were not checked. — [arXiv](https://arxiv.org/pdf/2603.12936)
- **Interactive-Scene-Reconstruction (hmz-15)** is older. It builds a contact graph from RGB-D streams, replaces objects with articulated CAD models, and exports URDF for ROS simulators. — [repo](https://github.com/hmz-15/Interactive-Scene-Reconstruction)

### Inferences
- For a 4–6 week MuJoCo project, the realistic options rank roughly as follows:
  - **(a) Hybrid manual pipeline.** Phone/LiDAR scan for the static shell and dimensions, plus a hand-authored MJCF drawer (box primitives and a slide joint), plus an image-to-3D or scanned cup with CoACD collision. Lowest risk.
  - **(b) LiteReality-Agent.** Very new and the only tool found that does phone scan → articulated MuJoCo room end-to-end. It is promising, but it depends on several paid or online services (OpenAI/Gemini, Modal, codex and claude CLIs) and on Blender 5.x, and it is only weeks old.
  - **(c) EmbodiedGen.** Useful for per-object assets with an MJCF converter and VLM physics.
  - **(d) URDFormer or Articulate-Anything.** Can produce a cabinet URDF from a photo for conversion to MJCF. Geometry is approximate, which matters for clearances.
- DRAWER, iTACO, RialTo and ACDC are the strongest research references for articulated kitchens. Their simulator lock-in (Isaac Sim, OmniGibson), heavy dependencies (iTACO's 48 GB NKSR, old torch versions) and lack of MJCF export make them poor bets for a quick MuJoCo demo. Cite them as related work instead.
- Superseded or older items:
  - URDFormer's environment (torch 1.12, cu113, 2024) is unlikely to install cleanly on Blackwell GPUs, which need CUDA 12.8+ builds. This is inferred from EmbodiedGen recommending cu128 for Blackwell.
  - Ditto (2022) and PARIS (2023) are superseded by video/RGB-D methods such as iTACO and ArtiTwinSplat, but remain useful baselines.
- Background knowledge, not re-verified this session: Ditto takes point clouds before and after an interaction; PARIS takes multi-view RGB of two articulation states and optimizes a NeRF-style part model.

### Gaps
- Could not verify the license or a clean Blackwell install for most research repos (Real2Code, Articulate-AnyMesh, SplatSim and iTACO have no license file in the API).
- ArtiTwinSplat's export format and code release are unconfirmed.
- Agentic Real2Sim (2607.19190) was found only via alphaxiv, and its ID or date looks unusual.
- The LiteReality-Agent README says its codex CLI defaults to "GPT-6". That is reported as the README states it and was not independently verified.
- No independent evaluation of LiteReality-Agent's articulation accuracy was found.

## How articulation (joint type, axis, limits) is estimated for drawers and cabinets

### Takeaway
There are four approaches:
1. **Retrieve a pre-articulated asset** whose joints are already known (ACDC, Articulate-Anything, EmbodiedGen/DIPO).
2. **Predict joint structure from a single image** with a learned model or LLM (URDFormer, Real2Code).
3. **Infer joints from observed motion** across video frames or two states, refined by geometric optimization (iTACO, Ditto, PARIS, ArtiTwinSplat).
4. **Detect the joint type and hinge in a static scene video** (DRAWER).

Joint limits are the least reliably estimated quantity. For a drawer, the axis is almost always the cabinet-front normal (prismatic), so measuring it by hand is trivial and more accurate than any method.

### Cited Findings
- ACDC: GPT decides whether an object is articulated. Articulated objects are matched only to articulated BEHAVIOR assets. Open/close skills use ground-truth joint info from the asset, with the handle found by ray-casting and the trajectory computed from joint type (prismatic or revolute). — [ACDC](https://arxiv.org/html/2410.07408v2)
- Articulate-Anything: a VLM actor-critic iteratively writes and corrects link and joint code (`articulate_link.py`, `articulate_joint.py`, `actor_critic.py`), compiled to URDF. Templates are retrieved from PartNet-Mobility by visual similarity. — [repo](https://github.com/vlongle/articulate-anything)
- Real2Code: oriented bounding boxes for parts feed a fine-tuned CodeLlama, which outputs joint code. — [Real2Code](https://arxiv.org/html/2406.08474v2); [README](https://github.com/MandiZhao/real2code)
- URDFormer: predicts URDF structure from image boxes (GroundingDINO) using a model trained on synthetic image→URDF pairs. — [README](https://github.com/WEIRDLabUW/urdformer); [secondary review](https://www.themoonlight.io/de/review/urdformer-a-pipeline-for-constructing-articulated-simulation-environments-from-real-world-images)
- iTACO: a MonST3R motion map and AutoSeg-SAM2 parts give a coarse joint prediction. Chamfer-loss refinement then optimizes "joint parameters of a single type of joint", run once per type, and the best fit is chosen. — [video2articulation](https://github.com/3dlg-hcvc/video2articulation)
- DRAWER: the articulation module identifies articulation types and hinge positions in a static-scene video. — [DRAWER](https://arxiv.org/html/2504.15278v2)
- ArtiTwinSplat: recovers "part structure and joint kinematics from observed motion alone" from RGB-D video. — [arXiv](https://arxiv.org/abs/2606.24628)
- URDF Synthesis from RGB-D Sequences (2606.18861) uses "Differentiable Joint Inference and Energy-Consistent Verification". This is the title-level claim only. — [arXiv](https://arxiv.org/pdf/2606.18861)

### Inferences
- Recommended path for the course drawer:
  - Model it as a `slide` joint whose axis is the outward normal of the cabinet face. Estimate that normal from the scan by fitting a plane to the cabinet front, or measure it.
  - Set `range` to the measured max extension, with a tape measure or by scanning open and closed states.
  - Add `damping` and `frictionloss` so the drawer doesn't glide freely.
- An optional "research" component: record an RGB-D video of the drawer opening (Record3D) and fit the prismatic axis from the motion of tracked points, as iTACO and Ditto do in principle. Report the angular error against the hand-measured axis.

### Gaps
- No source found gave quantitative axis or limit errors for the drawer category specifically in the 2025–2026 methods checked. The papers' tables were not opened.

## How physical properties (mass, friction, inertia) are estimated or assigned

### Takeaway
The 2025–2026 consensus pattern is **VLM priors, then optional refinement by interaction or system ID, then simulator defaults for anything left over**. VLMs (GPT-4o, Gemini 2.5 Pro) give plausible density, mass and friction guesses. Several papers stress that dynamics "cannot be reliably inferred from purely visual observations" and refine them via robot interaction or differentiable simulation. For a demo, the usual approach is: compute inertia from mesh geometry with an assumed density, and set friction to a literature or VLM value.

### Cited Findings
- EmbodiedGen assigns "VLM-inferred physical properties (scale, mass, friction)" with automatic quality-check retries. — [EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen)
- Phys2Real (arXiv 2510.11689) uses a VLM prior (for example on center of mass), builds meshes from segmented Gaussian splats, and fuses VLM priors with interaction-based estimates in an uncertainty-aware way. — [Phys2Real](https://arxiv.org/pdf/2510.11689v2)
- RigPI (arXiv 2606.25212) seeds differentiable-sim parameter identification with Gemini 2.5 Pro estimates of density and static friction, from which mass, CoM and inertia are derived. — [RigPI](https://arxiv.org/pdf/2606.25212)
- CoRAL (arXiv 2605.02600) uses GPT-4o to estimate mass and friction from an image for simulation. — [CoRAL](https://arxiv.org/pdf/2605.02600)
- Real2Sim with VLM-generated behavior trees (arXiv 2601.08454) argues that dynamic properties "cannot be reliably inferred from purely visual observations". A Franka executes VLM-written behavior trees to measure them. — [arXiv 2601.08454](https://arxiv.org/html/2601.08454v1)
- End-to-End Real-to-Sim refines physical parameters jointly with geometry using differentiable MuJoCo physics. — [arXiv 2506.04120](https://arxiv.org/pdf/2506.04120)
- TwinAligner identifies rigid physics from robot–object interaction. — [TwinAligner](https://arxiv.org/html/2512.19390v1)
- LiteReality-Agent gives each object mass, inertia and friction, and each asset is "compiled and solver-gated". The method is not described. — [LiteReality-Agent](https://github.com/LiteReality/LiteReality-Agent/)
- MuJoCo defaults and mechanics:
  - Geom `friction` defaults to "1 0.005 0.0001" (sliding, torsional, rolling).
  - When two geoms have equal priority, the element-wise maximum friction is used and the maximum condim is used.
  - Default solref is "0.02 1" and default solimp is "0.9 0.95 0.001 0.5 2".
  - "The timeconst parameter should be at least two times larger than the simulation time step." — [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html); [MuJoCo modeling](https://mujoco.readthedocs.io/en/stable/modeling.html)
- MuJoCo mesh `inertia` options are convex, exact, legacy and shell. The default is "legacy", but "convex is recommended". "exact" "requires a well-oriented, watertight mesh and will error otherwise". "legacy" "leads to volume overcounting for non-convex meshes". — [MuJoCo XMLreference.rst](https://github.com/google-deepmind/mujoco/blob/main/doc/XMLreference.rst)

### Inferences
- Practical recipe:
  - **Cup**: measure its mass on a kitchen scale and set `mass` directly, letting MuJoCo distribute inertia from the geometry. Use `shell` inertia, or `exact` on a watertight mesh, because a cup is hollow; convex-hull inertia would overestimate it.
  - **Drawer and cabinet**: use the measured or estimated mass of the drawer box. Cabinet bodies can be static (worldbody) geoms.
  - **Friction**: about 0.5–1.0 sliding for ceramic on wood or laminate is a reasonable starting value. Optionally ask a VLM for a prior and document it as an assumption.
- A one-afternoon "system ID" addition that would strengthen the report: measure the drawer's pull force or slide-out behavior, or the cup's sliding angle on the drawer floor, and tune `frictionloss`, `damping` and `friction` to match.

### Gaps
- No source quantified VLM accuracy for friction coefficients. Some of the papers above report errors, but their tables were not opened.
- The friction range for ceramic on wood above comes from general engineering knowledge, not a fetched source.

## Making meshes simulation-ready and converting to MJCF

### Takeaway
MuJoCo collides with the **convex hull** of each mesh geom. A concave object like a cup or a drawer interior therefore needs either:
- **convex decomposition** into many convex mesh geoms (CoACD is the current standard and is integrated into obj2mjcf), or
- **primitive approximation**: boxes for drawer walls, and a ring of boxes or capsules for a cup.

Keep high-poly visual meshes separate (`contype=conaffinity=0`, group 2) from collision geoms. URDF loads natively into MuJoCo and can be saved back out as MJCF.

### Cited Findings
- "While any collection of triangles can be loaded as a mesh and rendered, collision detection works with the convex hull of the mesh." — [MuJoCo XMLreference.rst](https://github.com/google-deepmind/mujoco/blob/main/doc/XMLreference.rst). The modeling guide likewise says mesh geoms "are convexified for collision purposes". — [MuJoCo modeling](https://mujoco.readthedocs.io/en/stable/modeling.html)
- `maxhullvert` (default -1) caps the number of convex-hull vertices via qhull. — [XMLreference.rst](https://github.com/google-deepmind/mujoco/blob/main/doc/XMLreference.rst)
- URDF compiler defaults: `discardvisual` is "false" for MJCF and "true" for URDF, and it discards geoms with contype=conaffinity=0. `fusestatic` is likewise "true" for URDF. To keep visuals when importing a URDF, set `discardvisual="false"`. "MuJoCo also supports loading models from other formats such as URDF." — [XMLreference.rst](https://github.com/google-deepmind/mujoco/blob/main/doc/XMLreference.rst); [modeling](https://mujoco.readthedocs.io/en/stable/modeling.html)
- **CoACD** (SIGGRAPH 2022), `SarahWeiii/CoACD`, MIT, ~1185 stars, active through 2026-09.
  - Install with `pip install coacd`. The input mesh does not need to be manifold, and the `-t` threshold trades quality against part count.
  - 2026-04 news: a **real metric mode (`-rm`)** lets the concavity threshold be set in meters, which suits scanned meshes. Example: `coacd.run_coacd(mesh, threshold=0.01, real_metric=True)`.
  - 2025-09 news: companion **PaMO** library converts any mesh to low-poly, manifold, intersection-free form (CUDA) as preprocessing. — [CoACD README](https://github.com/SarahWeiii/CoACD)
- **obj2mjcf** (`kevinzakka/obj2mjcf`, MIT, ~385 stars, last push 2024-10). It "generates an MJCF XML file pre-filled with materials, meshes and geom elements" and "optionally generates a collision mesh by performing a convex decomposition of the OBJ using CoACD". It was used to process MuJoCo Menagerie meshes. Install with `pip install obj2mjcf`. — [obj2mjcf README](https://github.com/kevinzakka/obj2mjcf)
- EmbodiedGen ships `MeshtoMJCFConverter` and a convex collision proxy. — [EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen)
- ACDC de-penetrates objects and resolves x-y collisions after placement. LiteReality-Agent has a layout stage that "settles the noisy scan layout into a collision-free one" and a `--shake` test that "measures what moves". — [ACDC](https://arxiv.org/html/2410.07408v2); [LiteReality-Agent](https://github.com/LiteReality/LiteReality-Agent/)

### Inferences
- For the cup:
  1. Get a mesh (scan or image-to-3D).
  2. Fix the scale in meters and make it watertight (PaMO, or trimesh/MeshLab repair).
  3. Run CoACD with `real_metric=True` and a threshold of about 2–5 mm so the cavity stays open.
  4. Use obj2mjcf to emit the MJCF.
  5. Check that the decomposition did not fill the cup interior.
- For the drawer: use **box primitives** (bottom, four walls, front panel). They are exact, fast and stable, and clearances can be set to measured values. Hollow-container convex decomposition of a scanned drawer is error-prone.
- A LiteReality-style "shake test" is cheap to copy. Step the sim for 1–2 s with zero control and assert that no free body moves more than a few millimeters. This catches interpenetration from scan noise.
- Mesh-to-mesh convex contacts in MuJoCo can jitter on thin walls. Set an explicit `margin` and condim, and use a smaller timestep for the cup-in-drawer contact. This is an inference from the solref/timestep guidance.

### Gaps
- Did not find a maintained, general-purpose URDF→MJCF converter repo beyond MuJoCo's built-in URDF loader (load the URDF, then `mj_saveLastXML`) and the EmbodiedGen converter. Specific third-party converter repos were not verified because the GitHub search API hit its rate limit.

## Practical capture tools and foundation models

### Takeaway
For a student team, the proven low-cost capture stack is:
- an **iPhone Pro with LiDAR**: Polycam or Scaniverse for textured room or object meshes, Record3D for RGB-D video, or the LiteReality Scanner / RoomPlan for room layouts;
- the **Grounded-SAM-2** family for segmentation;
- **VGGT/MASt3R/DUSt3R** for pose-free multi-view geometry when there is no depth;
- **TRELLIS(.2) / Hunyuan3D-2.1 / SAM 3D Objects** for completing occluded single objects such as the cup;
- **FoundationPose** for registering a mesh into the scene or tracking it.

All of the main ones are released, most recently active, and should run on 96 GB Blackwell cards with CUDA 12.8+ builds.

### Cited Findings
- iTACO's real-data protocol: iPhone 12 Pro with **Record3D** for RGB-D interaction video and **Polycam** for object surface scans. iPhone depth is 192×256, upsampled with PromptDA. — [video2articulation](https://github.com/3dlg-hcvc/video2articulation)
- The LiteReality Scanner app (free, iPhone) captures RGB, depth and a RoomPlan `room.usdz`. — [LiteReality-Agent](https://github.com/LiteReality/LiteReality-Agent/)
- ACDC chose Depth-Anything-v2 monocular depth over a depth camera because it handles reflective surfaces better. This is relevant for glossy kitchen surfaces. — [ACDC](https://arxiv.org/html/2410.07408v2)
- Real2Code uses DUSt3R for multi-view pose-free RGB reconstruction. — [Real2Code README](https://github.com/MandiZhao/real2code)
- Repository status from the GitHub API, 2026-10-16:

  | Repo | Stars | License | Last push | Notes |
  |---|---|---|---|---|
  | `facebookresearch/sam2` | ~20.0k | Apache-2.0 | 2026-05 | |
  | `IDEA-Research/GroundingDINO` | ~10.7k | Apache-2.0 | 2024-08 | |
  | `IDEA-Research/Grounded-SAM-2` | ~3.8k | Apache-2.0 | 2025-11 | "Ground and Track Anything in Videos with Grounding DINO, Florence-2 and SAM 2" |
  | `NVlabs/FoundationPose` | ~3.6k | NVIDIA custom ("NOASSERTION") | 2026-04 | CVPR 2024 Highlight, 6D pose estimation and tracking of novel objects |
  | `NVlabs/BundleSDF` | ~1.4k | custom | 2026-05 | |
  | `facebookresearch/vggt` | ~14.5k | custom | 2026-05 | CVPR 2025 Best Paper |
  | `naver/mast3r` | ~3.1k | custom (non-commercial per Naver) | 2025-06 | |
  | `microsoft/TRELLIS` | ~14.0k | MIT | 2026-06 | |
  | `microsoft/TRELLIS.2` | ~12.0k | MIT | 2026-07 | |
  | `Tencent-Hunyuan/Hunyuan3D-2.1` | ~4.2k | Tencent community license | 2025-10 | PBR materials |

  Sources: [sam2](https://github.com/facebookresearch/sam2); [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2); [FoundationPose](https://github.com/NVlabs/FoundationPose); [vggt](https://github.com/facebookresearch/vggt); [mast3r](https://github.com/naver/mast3r); [TRELLIS](https://github.com/microsoft/TRELLIS); [TRELLIS.2](https://github.com/microsoft/TRELLIS.2); [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- **SAM 3D Objects** (Meta, released November 2025 with SAM 3) takes one image plus an object mask. It outputs a mesh, a textured model, Gaussian splats and the object pose, and handles occlusion and clutter. All sources found were secondary, and the license was not verified. — [LearnOpenCV](https://learnopencv.com/sam-3d/); [Roboflow blog](https://blog.roboflow.com/sam-3d/)
- EmbodiedGen recommends the `cu128` build for RTX 50-series/Blackwell, and `cu126` for 40-series. — [EmbodiedGen](https://github.com/HorizonRobotics/EmbodiedGen)

### Inferences
- Minimal capture plan:
  1. Polycam or Scaniverse LiDAR scan of the kitchen counter and cabinet, exported as OBJ/GLB in meters. Use it for the static shell and for measuring drawer dimensions.
  2. A separate close-up scan or photo of the cup, then TRELLIS.2, Hunyuan3D-2.1 or SAM 3D Objects to get a closed mesh. Rescale it to calipered dimensions, because image-to-3D output is scale-free.
  3. Optionally, a Record3D video of the drawer opening for articulation fitting.
- On Blackwell (sm_120), older repos pinned to torch 1.x or cu11x (URDFormer, GroundingDINO's original build) will likely need rebuilding with torch ≥2.7 and cu128. Budget time for this or use the Grounded-SAM-2 or Florence-2 variants. This is an inference and was not tested.
- Check licenses before redistributing. FoundationPose, VGGT, MASt3R and Hunyuan3D have non-MIT or custom terms, which is fine for coursework.

### Gaps
- Polycam and Scaniverse export capabilities and pricing in 2026 were not verified with primary sources.
- SAM 3D Objects' repo, license and GPU needs were not verified from Meta's primary page.

## What fidelity aspects matter most for sim-to-real in pick-and-place / articulated-object tasks

### Takeaway
The strongest direct evidence (ACDC) is that **exact visual or geometric replication matters less than preserving functional affordances plus diversity**. A single precise digital twin transferred at 25% on cabinet-door opening, while "digital cousins" transferred at 90–95%. Other work finds that **visual fidelity alone is insufficient**, and that **both rendering and dynamics fidelity** matter for trustworthy sim evaluation. For a cup-in-drawer task, the critical fidelity is functional geometry (drawer opening size, wall heights, cup diameter and height, clearances), joint axis and range, and contact stability. Texture is not critical.

### Cited Findings
- ACDC door opening, real success rates: Twin 25%, Twin+Domain Randomization 55%, Twin+Cousin 95%, Cousin 90%. In simulation, the twin policy scored 100% but transferred poorly. — [ACDC arXiv](https://arxiv.org/html/2410.07408v2); [ACDC abstract via search](https://arxiv.org/html/2410.07408v2)
- RialTo: RL fine-tuning in on-the-fly digital twins gave "over 67%" gains in policy robustness. — [RialTo](https://arxiv.org/html/2403.03949v3)
- "High visual fidelity does not by itself guarantee reliable manipulation learning". Residual lighting, shadow, camera, calibration and occlusion gaps remain, and policies trained on narrow reconstructed geometry may overfit to the specific mesh. — [Function-Preserving Data Generation, arXiv 2609.18293](https://arxiv.org/pdf/2609.18293)
- Ablations in Real-to-Sim Robot Policy Evaluation with Gaussian Splatting (soft-body) show "rendering and dynamics fidelity are both crucial" to trustworthy simulation-based evaluation. — [arXiv 2511.04665](https://arxiv.org/html/2511.04665v2)
- TwinAligner explicitly separates visual alignment from dynamic alignment for physics-aware real2sim2real. — [TwinAligner](https://arxiv.org/html/2512.19390v1)
- SplatSim achieves zero-shot sim2real for RGB policies by swapping simulator rendering for Gaussian splats while keeping PyBullet physics. This implies the visual gap was the main bottleneck for RGB policies on that task. — [SplatSim README](https://github.com/qureshinomaan/SplatSim)
- A practitioner blog claims a 40% gain from physically accurate assets. This could **not** be verified against a primary source and should not be cited. — [simready.artlabs.ai](https://simready.artlabs.ai/resources/sim-to-real-transfer-physics-accuracy)

### Inferences
- For the course task, the fidelity priorities are:
  1. Drawer interior dimensions and opening clearance, both depth when open and height under the counter. Measure these with calipers or a tape and encode them as exact box primitives.
  2. Prismatic axis and range.
  3. Cup bounding dimensions and a watertight, decomposed collision shape.
  4. Contact parameters, so the cup rests stably.
  5. Visual appearance, which matters only if a vision policy is trained.
- A cheap "scene variation" story consistent with ACDC: randomize drawer width, depth, height, cup size and friction within measured tolerances. This is "cousin"-style variation and directly supports the course's "Scene Variation" theme.
- To show clearances are preserved, report a table of real versus sim measurements (drawer inner W/D/H, open extension, cup D/H, counter-to-drawer gap) with error in mm.

### Gaps
- No controlled study was found that isolates geometric-clearance error versus success rate for drawer-placement tasks specifically.
- The quantitative ablation tables of 2609.18293 and 2511.04665 were not opened, so no numbers are given for them.
