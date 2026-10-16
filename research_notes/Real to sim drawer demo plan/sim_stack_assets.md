# Simulation Stack and Assets for a MuJoCo "Place a Cup in a Drawer" Demo

Scope: framework choice, robot models, drawer/cup assets, control/IK/planning, headless rendering, and LLM-agent program interfaces for a team on conda env `robot` (Python 3.11, MuJoCo 3.16.0, headless Linux, EGL, 2x RTX PRO 6000 Blackwell). Researched 2026-10-16.

**Local environment check (run directly on the team's server, 2026-10-16):**
- `robot` env: Python 3.11.17, mujoco 3.16.0, numpy 2.4.6; robosuite, mink, torch, mediapy and imageio are **not yet installed**.
- GPUs: 2x "NVIDIA RTX PRO 6000 Blackwell Server Edition", driver 590.48.01.
- PyPI, current as of today: mujoco 3.16.0 is the newest release (3.15 and 3.14 came before it); robosuite 1.5.2 is the newest; mink 1.3.0 is the newest.
- **A `pip install --dry-run robosuite==1.5.2` into this env resolves to `mink==0.0.5` and DOWNGRADES numpy 2.4.6 -> 1.26.4.** mujoco stays at 3.16.0. It also pulls numba 0.68, scipy 1.17.1, opencv-python 4.11, qpsolvers 4.13 and quadprog.

---

## 1. Framework options and trade-offs (incl. headless EGL + Blackwell support)

### Takeaway
For a demo that must run within days, the lowest-risk path is **plain MuJoCo 3.16 + Python, using Menagerie robots, mink for IK and a hand-built MJCF drawer**. It needs no extra simulator, no CUDA or PyTorch, and EGL already works. RoboCasa has the richest kitchen drawer/cabinet scenes but sits on robosuite master/v1.5 with ~10 GB of assets and pins old dependencies, so give it its **own conda env** rather than installing it into `robot`. ManiSkill3, Isaac Lab and mjlab are heavier, GPU-stack-dependent options and are not needed for a scripted single-environment demo.

### Cited Findings
**robosuite**
- The current major release is v1.5. It adds diverse embodiments, composite and whole-body controllers, more teleop devices, and photorealistic rendering (Isaac Sim integration) — [robosuite docs overview](https://robosuite.ai/docs/overview.html); [GitHub releases](https://github.com/ARISE-Initiative/robosuite/releases)
- The latest release is v1.5.2, uploaded to PyPI on 2025-12-24. Its metadata has `mujoco>=3.3.0`, **`mink==0.0.5` (exact pin)**, numpy>=1.13.3, numba>=0.49.1, scipy, qpsolvers[quadprog], opencv-python and pynput — [PyPI robosuite JSON](https://pypi.org/pypi/robosuite/json)
- The v1.5.2 notes cover a fix for mink IK with delta input, a base-frame option for whole-body IK, multi-camera support in the OpenCV renderer, and an adjustment to the domain-randomization wrapper "for MuJoCo versions other than 3.1.1" — [GitHub releases](https://github.com/ARISE-Initiative/robosuite/releases)
- In v1.4, robosuite switched from mujoco_py to DeepMind's pip `mujoco` bindings, and the OpenCV renderer replaced the onscreen MuJoCo renderer — [GitHub releases](https://github.com/ARISE-Initiative/robosuite/releases)
- Local dry-run: robosuite 1.5.2 forces numpy down to 1.26.4 via mink 0.0.5, which conflicts with the env's numpy 2.4.6. Latest mink on PyPI is 1.3.0 — local `pip install --dry-run` / `pip index versions` output (see header).

**RoboCasa**
- The README now uses `conda create -c conda-forge -n robocasa python=3.11`. It then has you clone and install **robosuite from the master branch** ("important: use the master branch!"), followed by `pip install -e .` for robocasa, `python -m robocasa.scripts.setup_macros` and `python -m robocasa.scripts.download_kitchen_assets` (**~10 GB**). If numba/numpy problems appear, it says to use `conda install -c numba numba=0.56.4` — [robocasa GitHub](https://github.com/robocasa/robocasa)
- RoboCasa365 (v1.0) was released 2026-02-18 with 365 tasks and 2,500+ kitchen scenes. v1.0.1 followed on 2026-05-12 and a subtask-annotation pipeline update on 2026-10-14. The README gives 3,200+ objects in one place and 2,500+ in another (an inconsistency in the README itself). Objects are Objaverse plus AI-generated ("aigen") — [robocasa GitHub](https://github.com/robocasa/robocasa)
- The README does not mention headless rendering. The Hugging Face LeRobot RoboCasa guide recommends `MUJOCO_GL=egl` on headless servers and installs robocasa with `--no-deps` to avoid dependency-pin conflicts — [LeRobot RoboCasa docs](https://huggingface.co/docs/lerobot/robocasa)
- An older v0.2 instruction set used Python 3.10 with robosuite v1.5 — [Hello Robot stretch-mujoco RoboCasa page](https://docs.hello-robot.com/0.3/stretch-mujoco/third_party/robocasa/)
- A third-party benchmark repo notes that RoboCasa needs a nightly/fork robosuite while LIBERO needs robosuite 1.4.0, so the two need separate envs — [sim-env repo (via search)](https://github.com/Chaoqi-LIU/sim-env)

**ManiSkill3 (SAPIEN)**
- Install with `pip install --upgrade mani_skill torch` (nightly: `mani_skill-nightly`). Rendering needs a working **Vulkan** driver (`libvulkan1`, `/usr/share/vulkan/icd.d/nvidia_icd.json`, `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`). GPU simulation is Linux/NVIDIA only. The docs advise against "server" driver variants and runfile installs — [ManiSkill installation docs](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- ManiSkill notes that the original PartNet-Mobility data has annotation errors and that its automatic VHACD collision decomposition does not handle all cases — [ManiSkill paper](https://arxiv.org/pdf/2107.14483)
- I found **no ManiSkill/SAPIEN-specific Blackwell reports**, either positive or negative. The only related item is an Isaac Sim 5.1 Vulkan GUI crash on a Blackwell laptop that was fixed by a driver downgrade; headless mode was fine there — [NVIDIA forum](https://forums.developer.nvidia.com/t/isaac-sim-5-1-gui-crash-access-violation-on-rtx-5070-ti-blackwell-fixed-by-driver-downgrade-to-591-74/365335)

**Isaac Sim / Isaac Lab**
- The Isaac Sim 5.0 requirements list the RTX PRO 6000 Blackwell as the "ideal" GPU — [Isaac Sim 5.0 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html)
- The Isaac Sim 5.1 requirements list the RTX PRO 6000 Blackwell with Linux driver 580.65.06 — [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- The latest (6.x) docs list Linux driver 595.58.03 and deprecate Ubuntu 22.04 — [Isaac Sim latest requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- Isaac Sim 4.5 does **not** support Blackwell; a forum reply recommends 5.1 or 6.0 — [NVIDIA forum](https://forums.developer.nvidia.com/t/isaac-sim-4-5-gui-crashes-with-errors-when-running-isaaclab-examples/369827/3)
- An anecdotal report describes Isaac Sim 5.1 crashing on an RTX PRO 6000 under Windows Server 2025 with driver 595.97 — [NVIDIA forum](https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-crashes-shortly-after-startup-on-windows-server-2025-with-rtx-pro-6000-blackwell/370054)

**mjlab / MuJoCo Warp / Playground**
- mjlab combines Isaac Lab's manager-based API with MuJoCo Warp, a GPU MuJoCo maintained by Google DeepMind and NVIDIA as part of Newton and published on PyPI as `mujoco-warp`. Its README describes it as a **beta with "no stable release yet"** and warns of breaking changes. Training needs an NVIDIA GPU — [mjlab paper](https://arxiv.org/abs/2601.22074v1); [mujoco_warp GitHub](https://github.com/google-deepmind/mujoco_warp); [mjlab PyPI](https://pypi.org/project/mjlab/)
- The mjlab paper's v1 lists velocity-tracking, motion-imitation and manipulation reference tasks; v2 describes "three reference tasks". The two versions are inconsistent — [arXiv v1](https://arxiv.org/html/2601.22074v1), [arXiv v2](https://arxiv.org/pdf/2601.22074v2)
- MuJoCo Playground announced (beta) support for running on MuJoCo Warp — [Playground discussion #197](https://github.com/google-deepmind/mujoco_playground/discussions/197)

**PyTorch on Blackwell (relevant to ManiSkill / Isaac Lab / mjlab / Genesis / any learned policy)**
- sm_120 is supported since PyTorch 2.7.0 when built with CUDA 12.8+. Older cu121/cu126 builds fail with "no kernel image" — [PyTorch forum](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099); [PyTorch forum p.4](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099?page=4)

**MuJoCo itself**
- The Python bindings' version matches the MuJoCo C library version. PyPI classifiers list Python 3.10–3.15 — [PyPI mujoco](https://pypi.org/pypi/mujoco/json)
- The MuJoCo docs now also expose a **Filament** renderer (`mujoco.rendering.filament`, `mjrf`) next to the classic OpenGL renderer. A classic offscreen context is created with `mujoco.GLContext(w, h)` and can be current on only one thread at a time — [MuJoCo Python docs](https://mujoco.readthedocs.io/en/latest/python.html)

### Inferences
- Plain MuJoCo is the best fit for "running in days". The demo is a single environment with a scripted policy, so GPU-parallel simulators (MJX, Warp, ManiSkill GPU sim, Isaac Lab) add install risk on Blackwell without a payoff. Plain CPU MuJoCo with EGL rendering does not depend on the CUDA or PyTorch version at all.
- **Do not `pip install robosuite` into `robot` as-is**, because it would downgrade numpy to 1.26 and pin mink to 0.0.5. If robosuite or RoboCasa is wanted, for example to borrow RoboCasa kitchen fixtures, create a separate env such as `robocasa` (Python 3.11, robosuite master, robocasa) and let it pick its own pins. Use the `robot` env with the latest mink for the custom MuJoCo pipeline.
- Use RoboCasa as an **asset/scene source**, or as a second "benchmark-style" env, rather than as the core of the demo. Its kitchens with articulated drawers and cabinets are valuable for the "scene variation" part of the project.
- Isaac Lab is the most Blackwell-validated of the heavy stacks, given NVIDIA's own requirement tables. But it would mean moving off MuJoCo and adopting USD assets, which conflicts with the "MuJoCo-based" brief. The local driver (590.48.01) is newer than the 5.1 minimum (580.65.06) and older than the 6.x recommendation (595.58.03).
- mjlab and MuJoCo Warp become interesting later, if the team wants to batch-evaluate many generated scene variants on GPU. Treat them as beta.

### Gaps
- No direct confirmation was found that robosuite 1.5.2 or robocasa master runs correctly on **MuJoCo 3.16** specifically. The pin is only `mujoco>=3.3.0`, and MuJoCo has had several minor releases (3.4–3.16) with possible XML/API deprecations since then. **Needs a smoke test.**
- Genesis, LIBERO and MuJoCo Playground Blackwell/EGL status was not researched in depth. LIBERO is known (per the sim-env repo note above) to depend on old robosuite 1.4, which makes it a poor fit for MuJoCo 3.16.
- No ManiSkill-specific Blackwell compatibility information was found.

---

## 2. Robot models (MuJoCo Menagerie)

### Takeaway
Use the **Franka Panda** (`franka_emika_panda`: 7-DoF arm plus a 2-finger parallel gripper in one model) or the **FR3**, paired with Menagerie's `scene.xml`. It is the most common pick-and-place arm and has a ready mink example (`arm_panda.py`). **UR5e + Robotiq 2F-85** is the main alternative, with a longer reach. ALOHA is a bimanual setup and is overkill here.

### Cited Findings
- Menagerie includes these arms: Franka Emika Panda (9 DoF incl. gripper, Apache-2.0), Franka FR3 and FR3 v2 (7), UR5e and UR10e (6), Kinova Gen3 (7), xArm7 (13), Lite 6, KUKA iiwa 14, Sawyer, ViperX/WidowX, Trossen WXAI, PiPER, Flexiv Rizon4, ARX L5, SO101, Unitree Z1, and more. ALOHA is the dual-arm option. Gripper models include the Panda Gripper (2), Robotiq 2F-85 (8 DoF) and **Robotiq 2F-85 v4 (6)**, the xArm7 gripper, and the UMI gripper, plus dexterous hands — [mujoco_menagerie GitHub](https://github.com/google-deepmind/mujoco_menagerie)
- Load `scene.xml` for simulation, since `<model>.xml` contains only the robot. Each model's README states its minimum MuJoCo version. Not every model has an MJX variant (`*_mjx.xml`). Model quality varies, and an A+ to C grading system is planned but not yet applied — [mujoco_menagerie GitHub](https://github.com/google-deepmind/mujoco_menagerie)
- Models can be loaded with `python -m mujoco.viewer --mjcf ...` or through the Python package (`mm.load('<model>')`) — [mujoco_menagerie GitHub](https://github.com/google-deepmind/mujoco_menagerie)
- mink ships IK examples for the Panda (`arm_panda.py`, `dual_panda.py`), the UR5e (`arm_ur5e.py`), and also the iiwa14, ALOHA 2, xArm+LEAP, Kinova Gen3+LEAP, TidyBot and Stretch — [mink GitHub](https://github.com/kevinzakka/mink)

### Inferences
- The Panda's single-actuator, tendon-coupled parallel gripper is the simplest to script: one `ctrl` value for open or close. That makes it the easiest choice for an LLM-generated program.
- The Robotiq 2F-85 is a multi-DoF linkage. The existence of a separate "v4" model suggests the original needed fixes. If the team chooses the UR5e, prefer the v4 gripper and check its README.
- Grasp robustness on a cup: for a mug with a handle, grasp the rim or body from above or the side. Raise contact friction and use `condim` 4 or 6 (torsional friction) on the finger pads and the cup, or the cup tends to slip or rotate. This comes from general MuJoCo practice, not a cited source.
- For later real-to-sim work, match the arm to whatever physical robot the course has, if any (not stated in the brief).

### Gaps
- Gripper-specific modeling issues for Menagerie Panda vs. Robotiq were not documented in the sources I retrieved. Check each model's README and its changelog.

---

## 3. Assets: drawers/cabinets and cups/mugs

### Takeaway
**Cups:** `kevinzakka/mujoco_scanned_objects` (Google Scanned Objects as MJCF) is plug-and-play and includes several mugs and cups. **Drawers:** the fastest option is to author a simple prismatic-joint drawer in MJCF yourself, from box geoms in about 50 lines. For realistic kitchen drawers and scene variation, borrow RoboCasa fixtures. PartNet-Mobility has many cabinets with drawers, but they are URDF/SAPIEN-oriented and need conversion plus collision cleanup.

### Cited Findings
- `mujoco_scanned_objects` provides MJCF for **1,030** Google Scanned Objects. Each object has a visual `model.obj`, **32 V-HACD convex collision submeshes**, `texture.png` and `model.xml`, generated by obj2mjcf. Assets are CC-BY 4.0 and the XML is MIT — [mujoco_scanned_objects GitHub](https://github.com/kevinzakka/mujoco_scanned_objects)
- The repo's `models/` directory, listed via the GitHub API on 2026-10-16, contains mug/cup models including `ACE_Coffee_Mug_Kristen_16_oz_cup`, `Cole_Hardware_Mug_Classic_Blue`, `Room_Essentials_Mug_White_Yellow`, `Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White`, `Ecoforms_Cup_B4_SAN` and `Baby_Elements_Stacking_Cups` — [GitHub API listing](https://api.github.com/repos/kevinzakka/mujoco_scanned_objects/contents/models)
- RoboCasa provides 2,500+ procedurally composed kitchen scenes, with fixtures such as cabinets and drawers that tasks open and close, and thousands of Objaverse and AI-generated objects. The asset download is ~10 GB — [robocasa GitHub](https://github.com/robocasa/robocasa)
- PartNet-Mobility has articulated models but known annotation errors, and its automatic VHACD collision decomposition does not handle all cases — [ManiSkill paper](https://arxiv.org/pdf/2107.14483)
- I found no maintained PartNet-Mobility-to-MJCF converter. Generic URDF-to-MJCF helpers exist: the `mujoco_ros2_control` tool, which its authors call "highly experimental", and JAXSim's `ModelToMjcf` — [ros2_control tools](https://control.ros.org/kilted/doc/mujoco_ros2_control/mujoco_ros2_control/docs/tools.html); [JAXSim](https://jaxsim.readthedocs.io/stable/modules/mujoco.html)

### Inferences
- **Recommended for day 1:** a hand-written MJCF cabinet with one drawer: a body with a `slide` joint, a `range`, damping, and a handle geom. Pair it with a GSO mug. This avoids mesh collision problems: a drawer built from primitive boxes collides robustly, and a hollow drawer made from a single convex mesh would be "filled in" by MuJoCo's convex-hull collision.
- MuJoCo convex-hulls every mesh geom, so **concave containers such as drawers and cup interiors must be built from boxes or many convex pieces**. GSO's 32-piece decomposition handles the cup. A drawer interior needs separate box geoms for the bottom and four walls.
- Objaverse via RoboCasa, or generated assets, fit the later "many scene variants" goal. A procedural MJCF generator that randomizes drawer dimensions, positions, textures and the mug choice is the cheapest way to get variation.
- For reconstructed real scenes: import meshes as visual-only geoms, then add simplified primitive or convex-decomposed collision geoms with tools such as obj2mjcf or CoACD/V-HACD. The tool names come from the GSO repo's pipeline (obj2mjcf, V-HACD); CoACD is general knowledge.

### Gaps
- I did not verify whether RoboCasa fixture XMLs can be extracted cleanly and loaded standalone, outside robosuite's model-composition classes. They are generated by robosuite/robocasa Python code, so expect coupling.
- YCB-in-MuJoCo packages and Menagerie's lack of articulated furniture were not specifically checked.

---

## 4. Control: IK/OSC, motion planning, grasping and drawer manipulation

### Takeaway
Use **mink** (latest 1.3.0 in the `robot` env) for differential IK on the Panda. Run a waypoint-based scripted state machine on top of it: open drawer, approach the mug, grasp, lift, move over the drawer, lower, release, close drawer. Joint-position actuators track the IK solution. A full motion planner is not needed for a tabletop drawer scene. robosuite's OSC controllers are an alternative only if the team adopts robosuite.

### Cited Findings
- mink is "Python inverse kinematics based on MuJoCo" (Apache-2.0). It supports tasks in configuration or operational space, joint position and velocity limits, **collision avoidance between any geom pair**, and closed-chain constraints via equality constraints. Its Lie-group interface uses native C for hot paths. Install with `pip install mink`/`uv add mink` — [mink GitHub](https://github.com/kevinzakka/mink); [PyPI mink](https://pypi.org/project/mink)
- The latest mink on PyPI is 1.3.0, per local `pip index versions`. The GitHub citation block lists 1.1.0, so the repo metadata lags PyPI — [mink GitHub](https://github.com/kevinzakka/mink)
- robosuite integrates mink for IK, but pins `mink==0.0.5` and fixed mink delta-input IK in v1.5.2. robosuite also offers OSC position/pose controllers and composite whole-body controllers — [PyPI robosuite JSON](https://pypi.org/pypi/robosuite/json); [robosuite releases](https://github.com/ARISE-Initiative/robosuite/releases)
- PromptBook (a Code-as-Policies follow-up) reported 50–71% success at opening drawers and 100% at closing them on a real mobile manipulator with LLM-written primitives. Drawer opening is the hard step — [PromptBook NeurIPS'23 workshop](https://neurips.cc/virtual/2023/77272)

### Inferences
- **Scripted feasibility: high.** In simulation, the team has privileged state: drawer joint position, handle site pose and cup pose. A state machine using mink IK targets on `site`s (gripper TCP, drawer handle, cup grasp point, drawer-interior drop point) is straightforward.
- **Drawer opening tips:**
  - Grasp the handle, or hook behind a bar handle, and pull along the slide axis with IK targets updated from the drawer joint's qpos.
  - Alternatively, start with the drawer already open for v0 of the demo. Many papers do this.
  - Add a mink collision-avoidance task between the gripper/arm geoms and the cabinet geoms to avoid hitting the cabinet top when lowering the cup.
- **Motion planning:** cuRobo is CUDA/PyTorch-based, so Blackwell needs a cu128 PyTorch and a cuRobo build. mplib and OMPL are CPU options. All are optional, and none were verified here for MuJoCo 3.16 or Blackwell.
- Use **position actuators** (Menagerie's Panda uses position-controlled actuators with gains) and feed IK joint targets as `ctrl`. Avoid writing `qpos` directly except for teleport-style debugging.

### Gaps
- No sources were retrieved on cuRobo, mplib or OMPL compatibility with Blackwell or Python 3.11. Treat them as unverified.
- Exact mink 1.3.0 API changes vs. 0.0.x were not checked. Example scripts in the repo are the reference.

---

## 5. Headless rendering, recording, and remote viewing

### Takeaway
Set `MUJOCO_GL=egl` before importing mujoco. Render offscreen with `mujoco.Renderer(model, height, width)` → `update_scene(data, camera=...)` → `render()`, collect frames, and write MP4 with mediapy or imageio-ffmpeg. The interactive `mujoco.viewer` needs a display. On a headless server, use remote options: X forwarding/VNC, browser-based viewers (viser, Rerun) streaming poses, or simply saved MP4s.

### Cited Findings
- `mujoco.viewer.launch` blocks. `launch_passive` does not block, but the user script must call `sync()`. Both open an interactive window and so need a display. Classic offscreen rendering requires an OpenGL context (`mujoco.GLContext(w,h)`, `make_current()`), and a context can be current on only one thread at a time — [MuJoCo Python docs](https://mujoco.readthedocs.io/en/latest/python.html)
- MuJoCo also exposes a Filament renderer (`mujoco.rendering.filament.Renderer`) — [MuJoCo Python docs](https://mujoco.readthedocs.io/en/latest/python.html)
- `MUJOCO_GL=egl` is recommended for headless servers in the RoboCasa/LeRobot guide — [LeRobot RoboCasa docs](https://huggingface.co/docs/lerobot/robocasa)
- Menagerie models have live web previews on live.mujoco.org, a browser-based viewer — [mujoco_menagerie GitHub](https://github.com/google-deepmind/mujoco_menagerie)

### Inferences
- On a 2-GPU server, choose the EGL device with `MUJOCO_EGL_DEVICE_ID=0/1`, a MuJoCo env var known from general practice and not re-verified here. Keep one Renderer per process or thread, because of the GL context threading rule above.
- Recording recipe: `frames=[]`, step at control rate, render at about 30 fps from a fixed `<camera>` defined in the MJCF (front/side view of the drawer), then `mediapy.write_video('demo.mp4', frames, fps=30)`. `pip install mediapy` needs ffmpeg on PATH; conda-forge `ffmpeg` works.
- For an interactive look from a laptop, a lightweight viser or Rerun server that streams body poses and meshes is the most practical remote viewer. MJCF→viser/Rerun bridges are community code, unverified here. VNC/Xvfb with `mujoco.viewer` is the fallback.

### Gaps
- `mujoco.Renderer` and `MUJOCO_GL` documentation was not on the fetched Python docs page; the backend-selection docs live elsewhere. The steps above reflect established MuJoCo usage and should be smoke-tested. The user reports that EGL already works.
- Viser/Rerun MuJoCo integrations and MuJoCo WASM viewers were not researched.

---

## 6. Interfaces for an LLM coding agent to generate robot programs

### Takeaway
Expose a small, well-documented **Python skill API** in the Code-as-Policies style over the MuJoCo sim, and have the LLM write programs that compose these primitives. Candidate primitives:
- `get_object_pose(name)`
- `move_to(pose)`
- `grasp(name)`, `release()`
- `open_drawer(name)`, `close_drawer(name)`
- `place_in(container)`

Recent benchmarks show that LLM success depends strongly on how good these hand-crafted abstractions are.

### Cited Findings
- Code as Policies (Liang et al., 2022): LLMs write policy code that composes perception outputs and parameterized control-primitive APIs, prompted with few-shot examples. It was demonstrated on real robots — [CaP arXiv](https://arxiv.org/pdf/2209.07753)
- CaP-X (ICML 2026) has two parts:
  - CaP-Gym, an interactive environment where agents control robots by synthesizing programs over perception and control primitives.
  - CaP-Bench, with 7 simulation tasks and 12 models. Performance "improves with human-crafted abstractions but degrades as these priors are removed."

  The snippets did not identify the simulator — [CaP-X project page](https://rpl.cs.utexas.edu/publications/2026/07/19/fu-icml26-capx/); [ICML poster](https://icml.cc/virtual/2026/poster/66369)
- PromptBook targets writing new low-level primitives zero-shot instead of relying on given `pick`/`open drawer` primitives. On a real robot it reported 83% picking and 50–71% drawer-opening success — [PromptBook](https://neurips.cc/virtual/2023/77272)

### Inferences
- Implement the primitives once, on top of mink IK and the scripted controllers in Section 4, using sim ground-truth state. Swap in perception later when reconstructed scenes arrive.
- Give the LLM a sandboxed `run_program(code)` that executes, records an MP4, and returns a structured success check, e.g. "cup body inside drawer AABB, drawer closed, cup upright". This enables an agent loop of generate → run → read feedback → fix.
- Keep scene-agnostic names. Look up objects by semantic tags or site names in the MJCF, so the same program runs across generated scene variants.

### Gaps
- I could not confirm which simulator CaP-X/CaP-Gym uses, or whether its code and primitives could be reused directly with MuJoCo 3.16.
