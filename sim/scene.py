"""Kitchen scene for the cup-in-drawer task, built programmatically with MjSpec.

World frame: Panda base at the origin, +x points from the robot to the counter,
+z is up. The counter front face is at x = COUNTER_FRONT_X. The drawer slides
out of the counter toward the robot (toward -x).
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_SCENE = ROOT / "third_party/mujoco_menagerie/franka_emika_panda/scene.xml"
CUP_MODELS = ROOT / "third_party/mujoco_scanned_objects/models"

# Counter: a cabinet with a top plate and one drawer cavity (metres).
COUNTER_FRONT_X = 0.55
COUNTER_DEPTH = 0.40
COUNTER_HALF_Y = 0.45
COUNTER_TOP_Z = 0.30
TOP_THICKNESS = 0.03
CAVITY_Y = (-0.34, -0.02)
CAVITY_Z = (0.10, COUNTER_TOP_Z - TOP_THICKNESS)

# Drawer, expressed in the drawer body frame (origin at front-bottom-centre).
DRAWER_Y = -0.18
DRAWER_Z = 0.105
DRAWER_DEPTH = 0.36
DRAWER_HALF_WIDTH = 0.155
DRAWER_WALL_HEIGHT = 0.12
DRAWER_FRONT_HEIGHT = 0.16
DRAWER_PANEL = 0.02  # front panel thickness
DRAWER_WALL = 0.005  # side/back/bottom thickness
DRAWER_MAX_OPEN = 0.25
HANDLE_OFFSET_X = -0.045  # handle bar centre, in front of the panel
HANDLE_Z = (0.065, 0.15)  # handle bar extent above the drawer bottom

# Panda home pose (7 arm joints), from the Menagerie "home" keyframe.
HOME_ARM_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TCP_OFFSET = 0.1034  # hand frame to fingertip-pad centre, along hand z
# Top-down gripper orientation whose fingers close along world y (wxyz).
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
INIT_KEY = "init"  # keyframe holding the scene's initial state


@dataclass
class SceneConfig:
    cup_model: str = "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White"
    cup_scale: float = 0.55
    cup_mass: float = 0.12  # kg
    # Offset from the mesh origin to the centre of the cup body (unscaled
    # mesh units); the handle makes the mesh asymmetric.
    cup_body_center: tuple = (0.017, -0.002)
    cup_xy: tuple = (0.68, 0.20)  # cup body centre on the counter top
    cup_yaw: float = 0.0  # 0 = handle points toward the robot (-x)
    drawer_open: float = 0.0  # initial opening (m)
    gripper_stiffness: float = 400.0  # N/m on the finger tendon (Menagerie: 100)


def _yaw_quat(yaw):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def _lookat_xyaxes(pos, target):
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.concatenate([right, up])


def _add_box(body, name, center, half, rgba, **kwargs):
    return body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(center),
        size=list(half),
        rgba=list(rgba),
        **kwargs,
    )


def _add_counter(world):
    wood = [0.55, 0.42, 0.30, 1.0]
    top = [0.85, 0.85, 0.82, 1.0]
    counter = world.add_body(name="counter")
    cx = COUNTER_FRONT_X + COUNTER_DEPTH / 2
    half_x = COUNTER_DEPTH / 2
    cav_lo, cav_hi = CAVITY_Y
    z_lo, z_hi = CAVITY_Z
    _add_box(counter, "counter_top", (cx, 0, COUNTER_TOP_Z - TOP_THICKNESS / 2),
             (half_x, COUNTER_HALF_Y, TOP_THICKNESS / 2), top)
    _add_box(counter, "counter_base", (cx, 0, z_lo / 2),
             (half_x, COUNTER_HALF_Y, z_lo / 2), wood)
    side_z, side_hz = (z_lo + z_hi) / 2, (z_hi - z_lo) / 2
    right_w = cav_lo + COUNTER_HALF_Y
    _add_box(counter, "counter_right", (cx, -COUNTER_HALF_Y + right_w / 2, side_z),
             (half_x, right_w / 2, side_hz), wood)
    left_w = COUNTER_HALF_Y - cav_hi
    _add_box(counter, "counter_left", (cx, cav_hi + left_w / 2, side_z),
             (half_x, left_w / 2, side_hz), wood)
    _add_box(counter, "counter_back", (COUNTER_FRONT_X + COUNTER_DEPTH - 0.01, (cav_lo + cav_hi) / 2, side_z),
             (0.01, (cav_hi - cav_lo) / 2, side_hz), wood)


def _add_drawer(world, opening):
    panel_rgba = [0.62, 0.48, 0.34, 1.0]
    inner_rgba = [0.75, 0.63, 0.50, 1.0]
    drawer = world.add_body(name="drawer", pos=[COUNTER_FRONT_X, DRAWER_Y, DRAWER_Z])
    drawer.add_joint(
        name="drawer_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[-1, 0, 0],
        range=[0, DRAWER_MAX_OPEN],
        damping=5.0,
        frictionloss=0.5,
        ref=0.0,
    )
    hw, d, wall = DRAWER_HALF_WIDTH, DRAWER_DEPTH, DRAWER_WALL
    box_len = d - DRAWER_PANEL
    box_cx = DRAWER_PANEL + box_len / 2
    _add_box(drawer, "drawer_front", (DRAWER_PANEL / 2, 0, DRAWER_FRONT_HEIGHT / 2),
             (DRAWER_PANEL / 2, hw, DRAWER_FRONT_HEIGHT / 2), panel_rgba, mass=0.4)
    _add_box(drawer, "drawer_bottom", (box_cx, 0, wall), (box_len / 2, hw, wall), inner_rgba, mass=0.3)
    for side, sign in (("left", 1), ("right", -1)):
        _add_box(drawer, f"drawer_{side}", (box_cx, sign * (hw - wall), DRAWER_WALL_HEIGHT / 2),
                 (box_len / 2, wall, DRAWER_WALL_HEIGHT / 2), inner_rgba, mass=0.1)
    _add_box(drawer, "drawer_back", (d - wall, 0, DRAWER_WALL_HEIGHT / 2),
             (wall, hw, DRAWER_WALL_HEIGHT / 2), inner_rgba, mass=0.1)
    # Vertical bar handle on a single standoff, so a top-down gripper can pinch it.
    metal = [0.7, 0.7, 0.72, 1.0]
    z_lo, z_hi = HANDLE_Z
    _add_box(drawer, "handle_bar", (HANDLE_OFFSET_X, 0, (z_lo + z_hi) / 2),
             (0.008, 0.008, (z_hi - z_lo) / 2), metal, mass=0.05, friction=[1.5, 0.01, 0.001])
    standoff_len = -HANDLE_OFFSET_X
    _add_box(drawer, "handle_standoff", (-standoff_len / 2, 0, z_lo + 0.006),
             (standoff_len / 2, 0.006, 0.006), metal, mass=0.02)
    drawer.add_site(name="drawer_interior", pos=[box_cx, 0, wall * 2], size=[0.01] * 3,
                    rgba=[0, 0, 0, 0])
    return drawer


def _absolutize_assets(spec):
    """Rewrite asset paths as absolute, so specs loaded from different folders can be merged."""
    base = Path(spec.modelfiledir)
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str(base / spec.meshdir / mesh.file)
    for tex in spec.textures:
        if tex.file:
            tex.file = str(base / spec.texturedir / tex.file)
    spec.meshdir = ""
    spec.texturedir = ""


def _cup_spec(cfg):
    spec = mujoco.MjSpec.from_file(str(CUP_MODELS / cfg.cup_model / "model.xml"))
    _absolutize_assets(spec)
    for mesh in spec.meshes:
        mesh.scale = [cfg.cup_scale] * 3
    body = spec.body("model")
    for geom in body.geoms:
        if geom.contype:  # collision geoms (visual mesh has contype 0)
            geom.friction = [1.5, 0.01, 0.001]
            geom.condim = 4
        else:
            geom.density = 0  # visual mesh must not add mass
    return spec


def _assemble(cfg: SceneConfig) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
    _absolutize_assets(spec)
    for key in list(spec.keys):  # Menagerie keyframe no longer matches nq
        spec.delete(key)
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10

    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005] * 3,
                               rgba=[1, 0, 0, 0])
    grip = spec.actuator("actuator8")
    k = cfg.gripper_stiffness
    grip.gainprm[0] = 0.04 * k / 255
    grip.biasprm[1] = -k
    grip.biasprm[2] = -k / 10

    world = spec.worldbody
    world.add_light(pos=[0.4, -1.0, 2.0], dir=[0, 0.4, -1], diffuse=[0.5, 0.5, 0.5])
    world.add_camera(name="overview", pos=[-0.45, -1.15, 1.0],
                     xyaxes=_lookat_xyaxes([-0.45, -1.15, 1.0], [0.5, -0.02, 0.2]), fovy=45)
    world.add_camera(name="side", pos=[0.45, -1.35, 0.55],
                     xyaxes=_lookat_xyaxes([0.45, -1.35, 0.55], [0.45, -0.1, 0.22]), fovy=45)
    _add_counter(world)
    _add_drawer(world, cfg.drawer_open)

    cup_spec = _cup_spec(cfg)
    # Place the mesh origin so that the cup *body* centre lands on cfg.cup_xy.
    yaw = cfg.cup_yaw
    off = np.array(cfg.cup_body_center) * cfg.cup_scale
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    origin_xy = np.array(cfg.cup_xy) - rot @ off
    frame = world.add_frame(pos=[*origin_xy, COUNTER_TOP_Z + 0.001], quat=_yaw_quat(yaw))
    cup = frame.attach_body(cup_spec.body("model"), "cup_", "")
    cup.add_freejoint(name="cup_free")
    cup.add_site(name="cup_center", pos=[*off, 0.03], size=[0.005] * 3, rgba=[0, 0, 0, 0])
    return spec


def build_spec(cfg: SceneConfig = SceneConfig()) -> mujoco.MjSpec:
    """Full scene spec: cup rescaled to cfg.cup_mass, initial state stored as keyframe "init"."""
    spec = _assemble(cfg)
    model = spec.compile()
    scale = cfg.cup_mass / model.body_mass[model.body("cup_model").id]
    for geom in spec.body("cup_model").geoms:
        if geom.contype:
            geom.density *= scale
    model = spec.compile()

    data = mujoco.MjData(model)
    data.qpos[:7] = HOME_ARM_Q
    data.qpos[7:9] = 0.04
    data.qpos[model.joint("drawer_slide").qposadr[0]] = cfg.drawer_open
    data.ctrl[:7] = HOME_ARM_Q
    data.ctrl[7] = 255
    spec.add_key(name=INIT_KEY, qpos=data.qpos.tolist(), ctrl=data.ctrl.tolist())
    return spec


def reset(model, data):
    mujoco.mj_resetDataKeyframe(model, data, model.key(INIT_KEY).id)
    mujoco.mj_forward(model, data)


def build_model(cfg: SceneConfig = SceneConfig()):
    model = build_spec(cfg).compile()
    data = mujoco.MjData(model)
    reset(model, data)
    return model, data


def load_model(xml_path):
    """Load an exported scene and reset it to its initial state."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    reset(model, data)
    return model, data


def export_xml(spec: mujoco.MjSpec, xml_path, asset_dir):
    """Write a standalone MJCF; its meshes and textures are copied into a shared asset_dir."""
    xml_path, asset_dir = Path(xml_path), Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    spec.compile()  # to_xml needs a compiled spec; asset paths are still absolute here
    xml = spec.to_xml()
    for asset in [*spec.meshes, *spec.textures]:
        if not asset.file:
            continue
        src = Path(asset.file)
        dst = asset_dir / f"{src.parent.name}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        xml = xml.replace(f'file="{src}"', f'file="{os.path.relpath(dst, xml_path.parent)}"')
    xml_path.write_text(xml)
