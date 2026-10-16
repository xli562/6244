# Cup in a drawer

A standalone MuJoCo simulation with a Franka Panda arm and gripper, kitchen
counter, passive sliding drawer, and hollow mug. The robot opens the drawer,
grasps the cup, places it inside, and closes the drawer through physical contacts.
`waypoints.py` learns visual waypoints from calibrated RGBD and expert labels;
robot kinematics execute a programmed task sequence. `simulation.py` contains the
scene, `expert.py` supplies robot control and the state-based teacher, and
`fetch_assets.py` downloads assets. `learning.py` retains the earlier joint-action
imitation baseline.

## Verify the saved results

Run these commands from this repository's directory. This workspace already has
the assets, trained checkpoints, and a prepared Python environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
```

On this shared server, choose an available GPU using `nvidia-smi`. The recorded
runs used physical GPU 3. Set both indices to your chosen GPU; a single-GPU machine
normally uses 0:

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=3
export MUJOCO_EGL_DEVICE_ID=3
python -c "import torch, mujoco; print('PyTorch:', torch.__version__, 'MuJoCo:', mujoco.__version__); print('CUDA available:', torch.cuda.is_available())"
```

For the CUDA commands below, the final line should print `CUDA available: True`.
The recorded environment used Python 3.12, MuJoCo 3.15.0, NumPy 2.5.3, and PyTorch
2.7.1 + CUDA 11.8 on an A100. The dependency files allow version ranges; replay
with the recorded versions/hardware for the closest numerical agreement.
`--device cpu` is also supported, but numerical differences can affect outcomes.

First, replay three test scenes and save videos:

```bash
python waypoints.py evaluate --checkpoint outputs/waypoint_scene_v2.pt \
    --episodes 3 --seed-start 10000 --randomization scene --device cuda --video \
    --output outputs/my_replay/scene.json
```

This runs the robot automatically, headlessly. Open
`outputs/my_replay/scene_seed10000.gif` in the IDE or an image viewer to watch the
first trial; the other two GIFs show inverted and sideways starts. GIF playback is
five times simulation speed. `scene.json` contains the freshly measured outcomes.
`python simulation.py` opens a separate interactive scene viewer with manual joint
controls; it does not run the learned policy.

To independently replay the full reported comparison, evaluate all three saved
checkpoints on the same 30 test scenes. These commands write new reports and keep
the original reports in `outputs/generalization_v2/`:

```bash
for variation in none cup scene; do
    python waypoints.py evaluate \
        --checkpoint "outputs/waypoint_${variation}_v2.pt" \
        --episodes 30 --seed-start 10000 --randomization scene --device cuda \
        --output "outputs/my_verification/${variation}.json"
done

python waypoints.py compare \
    --original-report outputs/my_verification/none.json \
    --cup-report outputs/my_verification/cup.json \
    --scene-report outputs/my_verification/scene.json \
    --output outputs/my_verification/summary.json
```

The recorded counts are `none: 0/30`, `cup: 7/30`, and `scene: 25/30`.
The comparison prints recomputed counts, checks matched test configurations and
training settings, and writes confidence intervals and orientation/handle
breakdowns. Every evaluation report includes per-seed success, failure reason,
ordered task progress, final physical status, configuration, and source hashes.
A success requires opening, lifting, placement, and closure in order, a stable cup
fully contained on the drawer floor, closure within 5 mm, and no robot–cup contact.
Changing log/progress code changes source hashes; compare actual configurations
and per-seed outcomes as well as totals when checking a replay.

Run the integration checks independently:

```bash
MUJOCO_GL=egl python -m unittest -v
```

The current suite contains 20 tests. Replaying these checkpoints verifies the
saved-model results; training new models independently is described under
[Reproduce training](#reproduce-training). Checkpoints, datasets, and original
result files are ignored by Git, so a fresh clone needs training or a copy of the
saved artifacts before these replay commands can work.

## Progress and logs

Asset fetching, headless simulation, expert execution, collection, data loading,
training, and evaluation show terminal progress bars with counts, elapsed time,
rate, and estimated time remaining. Evaluation also shows the current robot
stage; training shows epoch/batch progress and running loss. Asset progress counts
completed files, including cache hits. GIF encoding is included in trial runtime.

Bars use standard error and hide automatically when it is not a terminal. Add
`--no-progress` to disable them explicitly. For collection, training, and
evaluation, add `--verbose` to print detailed per-trial or per-epoch information
to standard output. For example, append `--no-progress --verbose` when capturing
a training log. Full JSON reports and training histories are still saved without
verbose output.

## Run

Use the existing project Conda environment, or a Python 3.10+ environment:

```bash
conda activate 6244
python -m pip install -r requirements.txt
python fetch_assets.py
python simulation.py
```

The asset download is about 35 MB, is pinned to source commits, and is cached in
`assets/vendor/`. Subsequent runs use local assets. The interactive viewer needs a
display connection. Its actuator controls move individual arm joints; **G** toggles
the gripper, **R** samples a new cup reset, and **D** toggles the drawer force demo.

For the server, render headlessly with EGL:

```bash
MUJOCO_GL=egl python simulation.py --headless --seed 7 --output outputs/baseline
```

Generate a scene variant and show the drawer open after three simulated seconds:

```bash
MUJOCO_GL=egl python simulation.py --headless --seed 7 --randomize-scene \
    --orientation sideways --drawer-demo --seconds 3 --output outputs/variant
```

`--drawer-demo` applies an external force to cycle the drawer over six seconds.
It is an articulation diagnostic. Use `expert.py` for execution through the arm
and gripper, and `waypoints.py` for the working RGBD imitation policy.
MuJoCo runs the physics on the CPU; EGL supports headless rendering on the A100.

Each output directory contains:

| File | Contents |
| --- | --- |
| `rgb.png` | 640 × 480 color observation |
| `depth.npy` | Aligned float32 depth in meters, along the optical axis |
| `rgbd.npz` | RGB, depth, camera intrinsics, and `camera_to_world` |
| `metadata.json` | Configuration, seed, initial pose class, diagnostics, MuJoCo version, and asset revisions |
| `scene.xml` | Generated MJCF with absolute paths to the local cached assets |

Camera coordinates use x right, y down, z forward, and a top-left image origin.
For a pixel `(u, v)` with depth `z`, its camera point is
`z * inv(intrinsics) @ [u, v, 1]`. Apply `camera_to_world` to get world coordinates.
Background depth is the renderer's far clipping distance. RGBD is ideal simulated
sensor output; sensor noise and missing depth can be added later. Depth rendering
disables multisampling: multisample depth introduced a roughly 3 mm reconstruction
bias at 320 × 240. The plane reconstruction regression check now requires 0.1 mm
accuracy. `observe(..., camera='workspace_rgbd')` selects a closer fixed camera for
learning; the default `rgbd` camera provides an overview. Each observation includes
the calibration for its selected camera.

## Reset and variation

The default drawer starts closed. Cup resets sample x/y position, yaw, and one of
three pose families: upright, inverted, or sideways. The cup is positioned using
its rotated collision geometry, then physically settled. A reset is accepted
only after at least 0.2 seconds of low velocity with countertop contact, the
requested pose family, and the complete footprint inside the countertop. The
reset's returned simulation time is zero. Invalid placements are retried or raise
an error.

`--randomize-scene` changes counter height, drawer width/depth/height, its clearance
below the counter, and handle shape. Use `--handle bar`, `--handle knob`, or
`--handle loop` to select a shape explicitly. Scene geometry is fixed for an
instance; `reset()` varies the cup. Construct another instance to change geometry.
`SceneConfig` also exposes initial drawer opening, travel, and cup scale.
Cup starts are sampled in a reachable region: x in [0.02, 0.24] m and y in
[0.12, 0.36] m. The robot sits on a stand in front of the left side of the counter;
the previous tabletop mount could not reach the fully open drawer handle.

```python
from simulation import KitchenSimulation, SceneConfig

sim = KitchenSimulation(SceneConfig(handle='loop', drawer_opening=0.1), seed=7)
try:
    observation = sim.observe()  # RGBD and calibration; no object-state input.
    control = sim.data.ctrl.copy()
    control[7] = 0  # Gripper: 0 closed, 255 open.
    sim.step(control, n_steps=10)  # Seven joint targets in radians, then gripper.
    diagnostics = sim.reset(seed=8, orientation='inverted')
finally:
    sim.close()
```

The default physics step is 2 ms; ten steps give a 50 Hz control interval.
`status()` exposes privileged state for debugging and future evaluation, including
cup containment and drawer closure. Its success flag requires the entire cup's
collision geometry inside the drawer, contact with its floor, and closure within
5 mm, with the robot no longer contacting the cup. It evaluates the final
configuration; the expert and learned-policy evaluators additionally track opening,
lifting, placement, and closure in order and require a stable cup at completion.

## Expert demonstrations and imitation learning

The project-local `.venv` is prepared with MuJoCo and CUDA PyTorch. Activate it:

```bash
source .venv/bin/activate
```

For another installation, install `requirements.txt` and a PyTorch build matching
the NVIDIA driver. This environment was tested with PyTorch 2.7.1 + CUDA 11.8:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

See the [official PyTorch installer](https://pytorch.org/get-started/locally/) for
other platforms. `requirements-learning.txt` lists the optional learning dependencies.
GPU access may require running outside the execution sandbox. On a shared server,
select a free GPU with `CUDA_VISIBLE_DEVICES` and the matching physical EGL device
with `MUJOCO_EGL_DEVICE_ID`.

Run the full task and save a preview GIF (shown at five times simulation speed):

```bash
MUJOCO_GL=egl python expert.py --seed 3 --orientation sideways --video \
    --output outputs/expert_task
```

The expert uses simulator object poses to generate demonstration targets. It uses
bounded Jacobian inverse kinematics, smooth joint commands, and physical gripper
contacts. It applies no object forces, teleports, or attachment constraints.
Gravity compensation adjusts the arm's existing actuator targets. The gripper
servo is tuned to grasp handles, with force limited to 40 N at the tendon actuator.
For sideways cups it grips near the base to avoid catching the open rim.

## RGBD waypoint imitation

The working policy combines a learned RGBD segmentation/waypoint network with
demonstration retrieval in 3D. It identifies cup and handle pixels, back-projects
their depth, aligns their point clouds with training examples, and transfers the
examples' grasp and placement waypoints. Depth also identifies the countertop so
uncertain cup-edge pixels on the support plane cannot distort the match.

Task sequencing, inverse kinematics, and smooth actuator control are programmed.
Joint and finger feedback permit up to three cup-grasp attempts. This is a hybrid
imitation baseline; task ordering is not learned. At execution, scene perception
uses RGBD and camera calibration. The controller uses robot feedback and its
kinematic/dynamic model. Simulator object poses, true masks, configuration values,
contact labels, and the teacher are excluded from decision inputs. They are used
for training supervision and evaluation metrics. The robot applies no external
object forces, teleports, or attachment constraints.

The prepared checkpoint is `outputs/waypoint_scene_v2.pt`. Replay three development
scenes, cycling upright, inverted, and sideways starts:

```bash
MUJOCO_GL=egl python waypoints.py evaluate --checkpoint outputs/waypoint_scene_v2.pt \
    --episodes 3 --seed-start 1000 --randomization scene --video \
    --output outputs/learned_replay.json
```

The controller completed **12/12 development scenes** at seeds 1000–1011. The
fixed-scene model also reproduced its training configuration (**1/1**), which the
earlier joint-action CNN could not do. Training artifacts and videos are local to
this workspace and ignored by Git; a fresh checkout requires collection/training.

On the 30 fresh test kitchens, with matched optimizer budgets:

| Training variation | Full-task success |
| --- | --- |
| One repeated scene | 0/30 (0%) |
| Random cup position/orientation | 7/30 (23.3%) |
| Random cup and kitchen geometry/handles | **25/30 (83.3%)** |

For full scene randomization, success was 9/10 upright, 9/10 inverted, and 7/10
sideways. Its Wilson 95% interval is 66.4–92.7%. Four failures occurred while
lowering the cup; one motion after opening a low drawer was unreachable. The
state-based teacher completed 24/30 on the same scenes. These are preliminary
simulation results; the teacher and learned controller use different execution
checks and recovery behavior.

Collect and train another model:

```bash
MUJOCO_GL=egl python waypoints.py collect --episodes 768 --seed-start 3000 \
    --randomization scene --output outputs/my_waypoints
python waypoints.py train --data outputs/my_waypoints --epochs 80 \
    --steps-per-epoch 40 --batch-size 16 --device cuda --output outputs/my_waypoint_policy.pt
MUJOCO_GL=egl python waypoints.py evaluate --checkpoint outputs/my_waypoint_policy.pt \
    --episodes 30 --seed-start 20000 --randomization scene \
    --output outputs/my_waypoint_evaluation.json
```

## Reproduce training

To repeat all three training conditions, use new dataset/output directories and
the original settings below. This takes longer than replaying the saved models.
Each model receives 768 snapshots and 3,200 optimizer steps. The `none` condition
repeats one unique configuration; `cup` and `scene` use distinct configurations.
The learned policy uses training seed 0; collection uses scene seeds 3000–3767
(only 3000 for `none`). Evaluation uses seeds 10000–10029 for reproduction:

```bash
for variation in none cup scene; do
    python waypoints.py collect --episodes 768 --seed-start 3000 \
        --randomization "$variation" \
        --output "outputs/my_retraining/${variation}_data"
    python waypoints.py train --data "outputs/my_retraining/${variation}_data" \
        --seed 0 --epochs 80 --steps-per-epoch 40 --batch-size 16 --lr 0.0003 \
        --device cuda --output "outputs/my_retraining/${variation}.pt"
    python waypoints.py evaluate --checkpoint "outputs/my_retraining/${variation}.pt" \
        --episodes 30 --seed-start 10000 --randomization scene --device cuda \
        --output "outputs/my_retraining/${variation}.json"
done

python waypoints.py compare \
    --original-report outputs/my_retraining/none.json \
    --cup-report outputs/my_retraining/cup.json \
    --scene-report outputs/my_retraining/scene.json \
    --output outputs/my_retraining/summary.json
```

The collection command refuses to overwrite an existing dataset. Choose another
directory if repeating this workflow. New training runs can differ numerically
across software/hardware; they are not guaranteed to recreate identical weights
or the exact saved success counts. For a new generalization experiment, choose a
fresh evaluation seed range, such as `--seed-start 20000`, and use the same range
for every condition.

Each training example is a physically settled scene with expert **waypoint labels**,
rather than a full trajectory. NPZ files contain RGBD, calibration, training-only
masks, and waypoint targets. The manifest records configuration, seed, orientation,
and source hashes. No trajectory-success filtering is applied to these snapshots.
Defaults use the closer camera at 384 × 288. Use a new dataset directory per run.

Collection modes support the randomization comparison: `none` repeats one upright
cup configuration, `cup` varies cup position/orientation with a fixed kitchen, and
`scene` also varies counter height, drawer dimensions/clearance, and handle shape.
Varied trials cycle all three cup orientations. The same asset instances and
appearance are used throughout.

Training holds out complete initial configurations, keeping repeated resets
together. Both network fitting and waypoint memory use only the training
partition. The one-configuration condition uses training-loss checkpoint selection
because an independent validation configuration does not exist. Other conditions
select on validation loss. Evaluation rejects training and validation seeds unless
`--allow-training-seeds` explicitly labels a reproduction check. `--decoder neural`
evaluates the direct waypoint head; the default `registered` decoder uses 3D
demonstration alignment. Training logs show direct-head errors; task rollouts
measure the complete registered policy.

The matched study uses 768 training snapshots per condition, one training seed,
80 epochs × 40 batches × 16 examples, and the same architecture, optimizer,
camera, decoder, and controller. `--steps-per-epoch` samples with replacement to
match optimizer work across conditions. Final evaluations share 30 fresh scene
seeds, 10000–10029, with ten starts per cup orientation. Results, configuration
records, failures, and randomized-policy GIFs are in `outputs/generalization_v2/`.

Create a paired summary with Wilson confidence intervals and orientation/handle
breakdowns:

```bash
python waypoints.py compare --original-report outputs/generalization_v2/none.json \
    --cup-report outputs/generalization_v2/cup.json \
    --scene-report outputs/generalization_v2/scene.json \
    --output outputs/generalization_v2/summary.json
```

The comparison checks matched training settings, identical test configurations,
and disjoint development seeds. These are small simulation experiments with known
assets and ideal depth. Repeat training seeds and broaden the test distribution
before drawing strong research conclusions. Real-world transfer remains unmeasured.

## Earlier joint-action baseline

`learning.py collect` saves full successful expert trajectories and retains failed
attempts in its manifest. `learning.py train` fits a CNN on RGBD and causal robot
history to predict joint/gripper action chunks; `learning.py evaluate` runs it
without a teacher fallback. This remains available for later policy comparisons:

```bash
MUJOCO_GL=egl python learning.py collect --episodes 24 --seed-start 100 \
    --randomization scene --output outputs/my_trajectories
python learning.py train --data outputs/my_trajectories --epochs 100 \
    --device cuda --output outputs/my_joint_policy.pt
```

The earlier collection saved 19/24 trajectories. Its history-based relative-joint
CNN achieved 0/6 development tasks and failed a single-demonstration reproduction
check with both relative and absolute targets. Those diagnostic runs, with unequal
training budgets, are retained in `outputs/experiment_summary.json`. They motivated
the waypoint policy; they are separate from the matched randomization study.

## Assets and modeling choices

| Component | Upstream source | Changes |
| --- | --- | --- |
| Arm and gripper | [MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/franka_emika_panda), Apache-2.0 | Stand mounting position, grasp reference site, reset posture, gripper servo/force tuning |
| Drawer | [RoboCasa drawer template](https://github.com/robocasa/robocasa/blob/a9e152e2eea16d342dac40e4577fb4c68b932854/robocasa/models/assets/fixtures/cabinets/drawer.xml), MIT repository license | Resize inner box, adjust slide direction/travel and physical clearances, add front and parameterized handles |
| Cup | [LIBERO custom mug](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01/notebooks/custom_assets/libero_mug), MIT repository license | Uniform scale 0.85, free joint, 0.25 kg inertia approximation, contact tuning; retain hollow wall/handle collision geometry |
| Counter and cabinet | Parameterized primitive boxes | Simple geometry allows independent drawer variations without importing a full kitchen framework |

The downloader preserves the upstream license files alongside each asset. Full
source revisions are in `fetch_assets.py`. No RoboCasa, LIBERO, Gymnasium, or
training framework installation is required.

No real scene capture was supplied. Dimensions, friction, mass, and camera
settings are simulation defaults, rather than reconstructed measurements.
Real-scene reconstruction, clutter, new object assets, sensor noise, and larger
evaluation suites can extend this baseline. Generalization to real configurations
remains to be measured.

## Validation

```bash
MUJOCO_GL=egl python -m unittest -v
```

The tests check stable starts and persistence across all three pose families,
seed reproducibility, passive drawer limits, cup support during closure,
robot/gripper actuation, scene/handle variations, and camera depth against
independent MuJoCo ray intersections. Additional integration checks cover the full
physical task in all three orientations, force/attachment restrictions, causal
history, training-data isolation, split integrity, and absence of expert fallback.
Waypoint checks cover metric depth precision, pose transfer with spurious support
pixels, execution while oracle queries are forbidden, and rejection of invalid
experimental comparisons.
Tests use `unittest`; learning checks require the optional PyTorch dependency.
The development environment uses MuJoCo 3.15.0.

The local tutorials informed the implementation: `1 Introduction.ipynb` for
contacts, stepping, and RGB/depth rendering; `2 terrain_ model editing.ipynb` for
procedural geometry; `5 Least Squares.ipynb` for Jacobian IK. `3 Rollout.ipynb` and
`6 MJX.ipynb` provide options for later
batched experiments and training.
