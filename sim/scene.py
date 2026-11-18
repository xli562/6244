"""Kitchen scene for the cup-in-drawer task, built programmatically with MjSpec.

World frame: Panda base at the origin, +x points from the robot to the counter,
+z is up. The counter front face is at x = cfg.counter_front_x. The drawer
slides out of the counter toward the robot (toward -x).

Every dimension lives in SceneConfig, so a scene is fully described by one
config (a plain dict / JSON object). Nothing outside this module needs the
dimensions: the compiled model carries named sites and numeric fields
(`drawer_interior`, `handle_grasp`, `cup_center`, `cup_dims`, ...) that the
skill API and the success check read back.
"""

import os
import shutil
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_SCENE = ROOT / "third_party/mujoco_menagerie/franka_emika_panda/scene.xml"
CUP_MODELS = ROOT / "third_party/mujoco_scanned_objects/models"

# Panda home pose (7 arm joints), from the Menagerie "home" keyframe.
HOME_ARM_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TCP_OFFSET = 0.1034  # hand frame to fingertip-pad centre, along hand z
# Top-down gripper orientation whose fingers close along world y (wxyz).
TOP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
INIT_KEY = "init"  # keyframe holding the scene's initial state
CAVITY_GAP = 0.005  # clearance between the drawer and the cabinet cavity


@dataclass(frozen=True)
class Clutter:
    """A loose distractor object. `xy` is in the world frame for where="counter"
    and in the drawer frame (origin front-bottom-centre, x into the cabinet)
    for where="drawer"."""
    name: str
    shape: str = "box"  # "box" or "cylinder"
    size: tuple = (0.03, 0.03, 0.03)  # box: half extents; cylinder: (radius, half height)
    xy: tuple = (0.7, 0.0)
    yaw: float = 0.0
    where: str = "counter"
    mass: float = 0.2


@dataclass(frozen=True)
class SceneConfig:
    # Counter / cabinet (metres).
    counter_front_x: float = 0.55
    counter_depth: float = 0.40
    counter_half_y: float = 0.45
    counter_top_z: float = 0.30
    top_thickness: float = 0.03
    # Drawer. The body frame origin is at the front-bottom-centre of the drawer.
    drawer_y: float = -0.18
    drawer_depth: float = 0.36  # front face to back wall, outer
    drawer_half_width: float = 0.155  # outer
    drawer_wall_height: float = 0.12
    drawer_front_height: float = 0.16
    drawer_panel: float = 0.02  # front panel thickness
    drawer_wall: float = 0.005  # side / back / bottom half thickness
    drawer_max_open: float = 0.25
    drawer_damping: float = 5.0
    drawer_frictionloss: float = 0.5
    handle_offset_x: float = -0.045  # handle bar centre, in front of the panel
    handle_z: tuple = (0.065, 0.15)  # handle bar extent above the drawer bottom
    # Cup.
    cup_model: str = "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White"
    cup_diameter: float = 0.05  # outer diameter of the cup body (the mesh is rescaled to it)
    cup_height: float = 0.063
    cup_mass: float = 0.12  # kg
    # Offset from the mesh origin to the centre of the cup body (unscaled
    # mesh units); the handle makes the mesh asymmetric.
    cup_body_center: tuple = (0.017, -0.002)
    cup_friction: float = 1.5
    # Initial state.
    cup_xy: tuple = (0.68, 0.20)  # cup body centre on the counter top
    cup_yaw: float = 0.0  # 0 = handle points toward the robot (-x)
    drawer_open: float = 0.0  # initial opening (m)
    clutter: tuple = ()
    # Robot.
    gripper_stiffness: float = 400.0  # N/m on the finger tendon (Menagerie: 100)

    # --- derived geometry -------------------------------------------------
    @property
    def drawer_z(self):
        """World z of the drawer body origin (its outer bottom)."""
        return self.counter_top_z - self.top_thickness - CAVITY_GAP - self.drawer_front_height

    @property
    def interior_half(self):
        """Half extents of the usable drawer interior (x, y, z)."""
        length = self.drawer_depth - self.drawer_panel - 2 * self.drawer_wall
        return (length / 2, self.drawer_half_width - 2 * self.drawer_wall,
                (self.drawer_wall_height - 2 * self.drawer_wall) / 2)

    @property
    def interior_center(self):
        """Centre of the drawer interior, in the drawer body frame."""
        hx, _, hz = self.interior_half
        return (self.drawer_panel + hx, 0.0, 2 * self.drawer_wall + hz)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["clutter"] = tuple(Clutter(**{**c, "size": tuple(c["size"]), "xy": tuple(c["xy"])})
                             for c in d.get("clutter", ()))
        for k in ("handle_z", "cup_body_center", "cup_xy"):
            if k in d:
                d[k] = tuple(d[k])
        known = {f.name for f in fields(SceneConfig)}
        return SceneConfig(**{k: v for k, v in d.items() if k in known})


def load_config(path) -> SceneConfig:
    """Scene config from a data file whose "scene" entry overrides the defaults."""
    import json
    return SceneConfig.from_dict(json.loads(Path(path).read_text())["scene"])


REAL_SCENE = Path(__file__).parent / "data/real_scene.json"


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


def _add_counter(world, cfg):
    wood = [0.55, 0.42, 0.30, 1.0]
    top = [0.85, 0.85, 0.82, 1.0]
    counter = world.add_body(name="counter")
    cx = cfg.counter_front_x + cfg.counter_depth / 2
    half_x = cfg.counter_depth / 2
    half_y = cfg.counter_half_y
    cav_lo = cfg.drawer_y - cfg.drawer_half_width - CAVITY_GAP
    cav_hi = cfg.drawer_y + cfg.drawer_half_width + CAVITY_GAP
    z_lo = cfg.drawer_z - CAVITY_GAP
    z_hi = cfg.counter_top_z - cfg.top_thickness
    _add_box(counter, "counter_top", (cx, 0, cfg.counter_top_z - cfg.top_thickness / 2),
             (half_x, half_y, cfg.top_thickness / 2), top)
    _add_box(counter, "counter_base", (cx, 0, z_lo / 2), (half_x, half_y, z_lo / 2), wood)
    side_z, side_hz = (z_lo + z_hi) / 2, (z_hi - z_lo) / 2
    right_w = cav_lo + half_y
    _add_box(counter, "counter_right", (cx, -half_y + right_w / 2, side_z),
             (half_x, right_w / 2, side_hz), wood)
    left_w = half_y - cav_hi
    _add_box(counter, "counter_left", (cx, cav_hi + left_w / 2, side_z),
             (half_x, left_w / 2, side_hz), wood)
    _add_box(counter, "counter_back",
             (cfg.counter_front_x + cfg.counter_depth - 0.01, (cav_lo + cav_hi) / 2, side_z),
             (0.01, (cav_hi - cav_lo) / 2, side_hz), wood)
    counter.add_site(name="counter_top", pos=[cx, 0, cfg.counter_top_z],
                     type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_x, half_y, 0.001],
                     rgba=[0, 0, 0, 0])


def _add_drawer(world, cfg):
    panel_rgba = [0.62, 0.48, 0.34, 1.0]
    inner_rgba = [0.75, 0.63, 0.50, 1.0]
    drawer = world.add_body(name="drawer", pos=[cfg.counter_front_x, cfg.drawer_y, cfg.drawer_z])
    drawer.add_joint(
        name="drawer_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[-1, 0, 0],
        range=[0, cfg.drawer_max_open],
        damping=cfg.drawer_damping,
        frictionloss=cfg.drawer_frictionloss,
        ref=0.0,
    )
    hw, d, wall, panel = cfg.drawer_half_width, cfg.drawer_depth, cfg.drawer_wall, cfg.drawer_panel
    wall_h, front_h = cfg.drawer_wall_height, cfg.drawer_front_height
    box_len = d - panel
    box_cx = panel + box_len / 2
    _add_box(drawer, "drawer_front", (panel / 2, 0, front_h / 2), (panel / 2, hw, front_h / 2),
             panel_rgba, mass=0.4)
    _add_box(drawer, "drawer_bottom", (box_cx, 0, wall), (box_len / 2, hw, wall), inner_rgba,
             mass=0.3)
    for side, sign in (("left", 1), ("right", -1)):
        _add_box(drawer, f"drawer_{side}", (box_cx, sign * (hw - wall), wall_h / 2),
                 (box_len / 2, wall, wall_h / 2), inner_rgba, mass=0.1)
    _add_box(drawer, "drawer_back", (d - wall, 0, wall_h / 2), (wall, hw, wall_h / 2), inner_rgba,
             mass=0.1)
    # Vertical bar handle on a single standoff, so a top-down gripper can pinch it.
    metal = [0.7, 0.7, 0.72, 1.0]
    z_lo, z_hi = cfg.handle_z
    hx = cfg.handle_offset_x
    _add_box(drawer, "handle_bar", (hx, 0, (z_lo + z_hi) / 2), (0.008, 0.008, (z_hi - z_lo) / 2),
             metal, mass=0.05, friction=[1.5, 0.01, 0.001])
    _add_box(drawer, "handle_standoff", (hx / 2, 0, z_lo + 0.006), (-hx / 2, 0.006, 0.006), metal,
             mass=0.02)
    # Grasp point: 3 cm below the top of the bar.
    drawer.add_site(name="handle_grasp", pos=[hx, 0, z_hi - 0.03], size=[0.004] * 3,
                    rgba=[0, 0, 0, 0])
    drawer.add_site(name="drawer_interior", pos=list(cfg.interior_center),
                    type=mujoco.mjtGeom.mjGEOM_BOX, size=list(cfg.interior_half),
                    rgba=[0, 0, 0, 0])
    return drawer


def _add_clutter(world, cfg):
    colors = [[0.80, 0.25, 0.25, 1], [0.25, 0.55, 0.80, 1], [0.30, 0.70, 0.35, 1],
              [0.85, 0.70, 0.20, 1], [0.60, 0.35, 0.75, 1], [0.90, 0.50, 0.20, 1]]
    for i, c in enumerate(cfg.clutter):
        half_h = c.size[2] if c.shape == "box" else c.size[1]
        if c.where == "drawer":
            x = cfg.counter_front_x - cfg.drawer_open + c.xy[0]
            y = cfg.drawer_y + c.xy[1]
            z = cfg.drawer_z + 2 * cfg.drawer_wall + half_h + 0.001
        else:
            x, y = c.xy
            z = cfg.counter_top_z + half_h + 0.001
        body = world.add_body(name=f"clutter_{c.name}", pos=[x, y, z], quat=_yaw_quat(c.yaw))
        body.add_freejoint(name=f"clutter_{c.name}_free")
        kind = mujoco.mjtGeom.mjGEOM_BOX if c.shape == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER
        size = list(c.size) if c.shape == "box" else [c.size[0], c.size[1], 0]
        body.add_geom(name=f"clutter_{c.name}", type=kind, size=size, mass=c.mass,
                      rgba=colors[i % len(colors)], friction=[0.8, 0.01, 0.001])


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


_UNIT_DIMS = {}


def _cup_unit_dims(cup_model):
    """(body diameter, height) of the scanned cup mesh at scale 1, measured once per model."""
    if cup_model not in _UNIT_DIMS:
        spec = mujoco.MjSpec.from_file(str(CUP_MODELS / cup_model / "model.xml"))
        _absolutize_assets(spec)
        radius, height, _ = _cup_dims(spec.compile(), "model")
        _UNIT_DIMS[cup_model] = (2 * radius, height)
    return _UNIT_DIMS[cup_model]


def _cup_scale(cfg):
    """(xy scale, z scale) that give the mesh the configured body diameter and height."""
    diameter, height = _cup_unit_dims(cfg.cup_model)
    return cfg.cup_diameter / diameter, cfg.cup_height / height


def _cup_spec(cfg):
    spec = mujoco.MjSpec.from_file(str(CUP_MODELS / cfg.cup_model / "model.xml"))
    _absolutize_assets(spec)
    sxy, sz = _cup_scale(cfg)
    for mesh in spec.meshes:
        mesh.scale = [sxy, sxy, sz]
    body = spec.body("model")
    for geom in body.geoms:
        if geom.contype:  # collision geoms (visual mesh has contype 0)
            geom.friction = [cfg.cup_friction, 0.01, 0.001]
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
    _add_counter(world, cfg)
    _add_drawer(world, cfg)
    _add_clutter(world, cfg)

    cup_spec = _cup_spec(cfg)
    # Place the mesh origin so that the cup *body* centre lands on cfg.cup_xy.
    yaw = cfg.cup_yaw
    off = np.array(cfg.cup_body_center) * _cup_scale(cfg)[0]
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    origin_xy = np.array(cfg.cup_xy) - rot @ off
    frame = world.add_frame(pos=[*origin_xy, cfg.counter_top_z + 0.001], quat=_yaw_quat(yaw))
    cup = frame.attach_body(cup_spec.body("model"), "cup_", "")
    cup.add_freejoint(name="cup_free")
    return spec


def _cup_dims(model, body_name="cup_model"):
    """(body radius, height, z of the bottom in the cup body frame), from the visual mesh."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = model.body(body_name)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == body.id and model.geom_contype[g] == 0:
            mid = model.geom_dataid[g]
            verts = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            local = (world - data.xpos[body.id]) @ data.xmat[body.id].reshape(3, 3)
            lo, hi = local.min(0), local.max(0)
            return float(min(hi[0] - lo[0], hi[1] - lo[1]) / 2), float(hi[2] - lo[2]), float(lo[2])
    raise ValueError("cup has no visual mesh")


def build_spec(cfg: SceneConfig = SceneConfig()) -> mujoco.MjSpec:
    """Full scene spec: cup rescaled to cfg.cup_mass, initial state stored as keyframe "init"."""
    spec = _assemble(cfg)
    model = spec.compile()
    scale = cfg.cup_mass / model.body_mass[model.body("cup_model").id]
    cup = spec.body("cup_model")
    for geom in cup.geoms:
        if geom.contype:
            geom.density *= scale
    radius, height, z_bottom = _cup_dims(model)
    off = np.array(cfg.cup_body_center) * _cup_scale(cfg)[0]
    cup.add_site(name="cup_center", pos=[*off, z_bottom + height / 2], size=[0.005] * 3,
                 rgba=[0, 0, 0, 0])
    spec.add_numeric(name="cup_dims", size=2, data=[radius, height])
    spec.add_numeric(name="handle_dims", size=2, data=[0.008, cfg.handle_z[1] - cfg.handle_z[0]])
    model = spec.compile()

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
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


__all__ = ["SceneConfig", "Clutter", "build_spec", "build_model", "load_model", "reset",
           "export_xml", "replace"]
