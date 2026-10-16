"""Standalone MuJoCo kitchen scene with stable resets and an RGBD camera."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from fetch_assets import ASSET_DIR, SOURCES
from progress import progress_bar


HOME = np.array([0, -0.45, 0, -2.1, 0, 1.65, 0.7854])
ORIENTATIONS = ('upright', 'inverted', 'sideways')


@dataclass
class SceneConfig:
    """Scene dimensions in meters; physical properties are simulation defaults."""

    counter_width:float = 1.2
    counter_depth:float = 0.65
    counter_height:float = 0.85
    drawer_width:float = 0.50
    drawer_depth:float = 0.48
    drawer_height:float = 0.18
    drawer_clearance:float = 0.10  # Countertop to top of drawer, in meters.
    drawer_travel:float = 0.38
    drawer_opening:float = 0.0
    handle:str = 'bar'
    cup_scale:float = 0.85
    timestep:float = 0.002

    def __post_init__(self):
        """Reject dimensions that would invalidate cabinet or drawer clearances."""
        dimensions = [value for value in asdict(self).values()
                      if isinstance(value, (float, int))]
        if not np.isfinite(dimensions).all():
            raise ValueError('Scene dimensions must be finite.')
        if not (1.1 <= self.counter_width <= 1.8
                and 0.55 <= self.counter_depth <= 0.85
                and 0.70 <= self.counter_height <= 1.1):
            raise ValueError('Counter dimensions exceed the supported workspace.')
        if not (0.35 <= self.drawer_width <= min(0.65, self.counter_width - 0.5)
                and 0.30 <= self.drawer_depth <= self.counter_depth - 0.08
                and 0.12 <= self.drawer_height <= 0.28
                and 0.07 <= self.drawer_clearance <= 0.16):
            raise ValueError('Drawer dimensions do not fit the cabinet.')
        if not (0 < self.drawer_travel < self.drawer_depth
                and 0 <= self.drawer_opening <= self.drawer_travel):
            raise ValueError('Drawer opening/travel must fit the drawer depth.')
        if self.handle not in ('bar', 'knob', 'loop'):
            raise ValueError('Handle must be bar, knob, or loop.')
        if not (0.6 <= self.cup_scale <= 1.0 and 0.0005 <= self.timestep <= 0.002):
            raise ValueError('Use cup_scale in [0.6, 1.0] and timestep in [.0005, .002].')

    @classmethod
    def randomized(cls, seed:int) -> 'SceneConfig':
        """Return a reproducible variant within the supported geometry ranges."""
        rng = np.random.default_rng(seed)
        return cls(counter_height=float(rng.uniform(0.78, 0.94)),
                   drawer_width=float(rng.uniform(0.40, 0.60)),
                   drawer_depth=float(rng.uniform(0.40, 0.55)),
                   drawer_height=float(rng.uniform(0.15, 0.24)),
                   drawer_clearance=float(rng.uniform(0.07, 0.13)),
                   handle=str(rng.choice(['bar', 'knob', 'loop'])))


def numbers(values) -> str:
    """Format a numeric sequence as an MJCF attribute."""
    return ' '.join(f'{value:.9g}' for value in values)


def add_box(parent:ET.Element, name:str, pos, size, **attributes) -> ET.Element:
    """Add a named box to an XML body and return the element."""
    return ET.SubElement(parent, 'geom', name=name, type='box', pos=numbers(pos),
                         size=numbers(size), **attributes)


def build_scene(config:SceneConfig) -> str:
    """Combine pinned upstream assets and configured geometry into MJCF text."""
    for filename in ['panda/panda.xml', 'drawer/drawer.xml', 'cup/libero_mug.xml']:
        if not (ASSET_DIR / filename).is_file():
            raise FileNotFoundError('Missing assets. Run: python fetch_assets.py')

    root = ET.parse(ASSET_DIR / 'panda/panda.xml').getroot()
    root.set('model', 'cup_in_drawer')
    root.remove(root.find('keyframe'))
    root.find('compiler').set('meshdir', str(ASSET_DIR / 'panda/assets'))
    root.find('option').set('timestep', str(config.timestep))
    gripper = root.find("actuator/general[@name='actuator8']")
    gripper.set('gainprm', str(0.04 * 2000 / 255))
    gripper.set('biasprm', '0 -2000 -30')
    gripper.set('forcerange', '-40 40')
    visual = ET.SubElement(root, 'visual')
    visual.append(ET.Element('global', offwidth='1280', offheight='960'))
    # Multisample depth resolves toward foreground samples, biasing metric XYZ.
    ET.SubElement(visual, 'quality', offsamples='0')
    ET.SubElement(root, 'statistic', center='0 0.15 0.65', extent='1.8')
    defaults = root.find('default')
    ET.SubElement(defaults, 'geom', friction='0.9 0.01 0.001',
                  solref='0.008 1', solimp='0.95 0.99 0.001')
    asset = root.find('asset')
    for name, rgba in [('cabinet', '.12 .25 .27 1'), ('wood', '.50 .30 .16 1'),
                       ('stone', '.85 .83 .76 1'), ('metal', '.55 .58 .61 1')]:
        ET.SubElement(asset, 'material', name=name, rgba=rgba, shininess='.2')
    ET.SubElement(asset, 'texture', name='floor_tex', type='2d', builtin='checker',
                  rgb1='.32 .34 .35', rgb2='.38 .40 .41', width='128', height='128')
    ET.SubElement(asset, 'texture', name='sky', type='skybox', builtin='gradient',
                  rgb1='.55 .65 .75', rgb2='.9 .92 .94', width='512', height='3072')
    ET.SubElement(asset, 'material', name='floor_mat', texture='floor_tex',
                  texrepeat='4 4', texuniform='true')

    world = root.find('worldbody')
    height, depth, width = (config.counter_height, config.counter_depth,
                            config.counter_width)
    mount_height = height - .12
    world.find("body[@name='link0']").set('pos', numbers([-.30, -.18, mount_height]))
    add_box(world, 'robot_stand', [-.30, -.18, mount_height / 2],
            [.075, .075, mount_height / 2], material='metal')
    add_box(world, 'robot_stand_base', [-.30, -.18, .025], [.17, .17, .025],
            material='metal')
    hand = world.find(".//body[@name='hand']")
    ET.SubElement(hand, 'site', name='grasp', pos='0 0 .103', size='.005',
                  rgba='0 0 0 0')
    ET.SubElement(world, 'light', pos='0 -.3 2.5', dir='0 .2 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'light', pos='1 .8 2', dir='-.3 -.2 -1',
                  diffuse='.5 .5 .5', castshadow='false')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 .1',
                  material='floor_mat')
    add_box(world, 'wall', [0, depth + .18, 1.1], [1.8, .04, 1.1],
            rgba='.8 .82 .79 1')
    add_box(world, 'countertop', [0, depth / 2, height - .025],
            [width / 2, depth / 2, .025], material='stone')
    for side in [-1, 1]:
        add_box(world, f'cabinet_side_{side}',
                [side * (width / 2 - .015), depth / 2, (height - .05) / 2],
                [.015, depth / 2, (height - .05) / 2], material='cabinet')
    add_box(world, 'cabinet_back', [0, depth - .015, (height - .05) / 2],
            [width / 2 - .03, .015, (height - .05) / 2], material='cabinet')
    add_box(world, 'cabinet_base', [0, depth / 2, .065],
            [width / 2 - .03, depth / 2 - .03, .015], material='cabinet')

    # Reuse RoboCasa's articulated inner box; its template needs sized clearances.
    drawer = ET.parse(ASSET_DIR / 'drawer/drawer.xml').find('.//body[@name="inner_box"]')
    drawer.set('name', 'drawer')
    drawer_z = height - config.drawer_clearance - config.drawer_height / 2
    drawer.set('pos', numbers([.14, .045 + config.drawer_depth / 2, drawer_z]))
    for element in list(drawer):
        if element.tag == 'site' or element.get('class') == 'region':
            drawer.remove(element)
    joint = drawer.find('joint')
    joint.attrib.update(name='drawer_slide', axis='0 -1 0', range=numbers([0, config.drawer_travel]),
                        damping='8', frictionloss='1.5', armature='.02')
    w, d, h = config.drawer_width / 2, config.drawer_depth / 2, config.drawer_height / 2
    geometry = {'inner_bottom': ([0, 0, -h + .006], [w, d, .006], '.8'),
                'inner_back': ([0, d - .006, .006], [w - .012, .006, h - .006], '.3'),
                'inner_left': ([-w + .006, 0, .006], [.006, d, h - .006], '.3'),
                'inner_right': ([w - .006, 0, .006], [.006, d, h - .006], '.3')}
    for name, (pos, size, mass) in geometry.items():
        geom = drawer.find(f"geom[@name='{name}']")
        geom.attrib.clear()
        geom.attrib.update(name=f'drawer_{name}', type='box', pos=numbers(pos),
                           size=numbers(size), mass=mass, material='wood')
    add_box(drawer, 'drawer_front', [0, -d - .009, 0], [w + .012, .009, h + .008],
            mass='.5', material='cabinet')
    handle_y = -d - .055
    ET.SubElement(drawer, 'site', name='drawer_handle', pos=numbers([0, handle_y, 0]),
                  size='.005', rgba='0 0 0 0')
    if config.handle == 'knob':
        ET.SubElement(drawer, 'geom', name='handle_stem', type='capsule',
                      fromto=numbers([0, -d - .018, 0, 0, handle_y, 0]),
                      size='.009', mass='.03', material='metal')
        ET.SubElement(drawer, 'geom', name='handle_knob', type='sphere',
                      pos=numbers([0, handle_y, 0]), size='.018', mass='.05', material='metal')
    else:
        for side in [-1, 1]:
            ET.SubElement(drawer, 'geom', name=f'handle_mount_{side}', type='capsule',
                          fromto=numbers([side * .07, -d - .018, 0,
                                          side * .07, handle_y, 0]),
                          size='.008', mass='.025', material='metal')
        if config.handle == 'bar':
            ET.SubElement(drawer, 'geom', name='handle_bar', type='capsule',
                          fromto=numbers([-.07, handle_y, 0, .07, handle_y, 0]),
                          size='.009', mass='.08', material='metal')
        else:
            for name, ends in [('left', [-.07, handle_y, 0, -.045, handle_y - .02, -.02]),
                               ('middle', [-.045, handle_y - .02, -.02, .045, handle_y - .02, -.02]),
                               ('right', [.045, handle_y - .02, -.02, .07, handle_y, 0])]:
                ET.SubElement(drawer, 'geom', name=f'handle_loop_{name}', type='capsule',
                              fromto=numbers(ends), size='.008', mass='.03', material='metal')
    world.append(drawer)

    lower_top = drawer_z - h - .025
    add_box(world, 'lower_panel', [0, .036, (.08 + lower_top) / 2],
            [width / 2 - .03, .009, (lower_top - .08) / 2], material='cabinet')
    upper_bottom, upper_top = drawer_z + h + .012, height - .05
    add_box(world, 'upper_panel', [0, .036, (upper_bottom + upper_top) / 2],
            [width / 2 - .03, .009, (upper_top - upper_bottom) / 2], material='cabinet')
    for name, left, right in [('left', -width / 2 + .03, .14 - w - .016),
                              ('right', .14 + w + .016, width / 2 - .03)]:
        add_box(world, f'front_panel_{name}', [(left + right) / 2, .036, drawer_z],
                [(right - left) / 2, .009, h + .008], material='cabinet')

    mug = ET.parse(ASSET_DIR / 'cup/libero_mug.xml').getroot()
    for element in mug.find('asset'):
        if 'file' in element.attrib:
            element.set('file', str(ASSET_DIR / 'cup' / element.get('file')))
        if element.tag == 'mesh':
            element.set('scale', numbers(np.array([.01] * 3) * config.cup_scale))
        asset.append(element)
    cup = mug.find('.//body[@name="object"]')
    cup.set('name', 'cup')
    ET.SubElement(cup, 'freejoint', name='cup_free')
    ET.SubElement(cup, 'inertial', pos=numbers([0, 0, .035 * config.cup_scale]),
                  mass='.25', diaginertia='.00025 .00025 .00020')
    for index, geom in enumerate(cup.iter('geom')):
        geom.set('name', f'cup_geom_{index}')
        geom.set('mass', '0')
        geom.attrib.pop('density', None)
        if geom.get('type') != 'mesh':
            geom.set('group', '3')
            geom.set('rgba', '0 0 0 0')
            geom.set('solref', '.008 1')
            geom.set('solimp', '.95 .99 .001')
            geom.set('condim', '4')
        for attribute in ['pos', 'size']:
            if attribute in geom.attrib:
                values = np.fromstring(geom.get(attribute), sep=' ') * config.cup_scale
                geom.set(attribute, numbers(values))
    for body in cup.iter('body'):
        if 'pos' in body.attrib:
            body.set('pos', numbers(np.fromstring(body.get('pos'), sep=' ') * config.cup_scale))
    world.append(cup)

    for name, position, target in [('rgbd', [1.25, -1.35, 1.65], [0, .15, height + .1]),
                                    ('workspace_rgbd', [.75, -.75, 1.45], [.10, .10, height - .05])]:
        camera_z = np.array(position) - target
        camera_z /= np.linalg.norm(camera_z)
        camera_x = np.cross([0, 0, 1], camera_z)
        camera_x /= np.linalg.norm(camera_x)
        camera_y = np.cross(camera_z, camera_x)
        ET.SubElement(world, 'camera', name=name, pos=numbers(position),
                      xyaxes=numbers(np.r_[camera_x, camera_y]), fovy='55')
    ET.indent(root)
    return ET.tostring(root, encoding='unicode')


class KitchenSimulation:
    def __init__(self, config:SceneConfig|None=None, seed:int=0,
                 orientation:str='random'):
        """Build a configured scene and settle a seeded cup pose on the counter."""
        self.config = config or SceneConfig()
        self.xml = build_scene(self.config)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.cup_id = self.model.body('cup').id
        self.cup_qpos = self.model.joint('cup_free').qposadr[0]
        self.cup_dof = self.model.joint('cup_free').dofadr[0]
        self.drawer_qpos = self.model.joint('drawer_slide').qposadr[0]
        self.drawer_dof = self.model.joint('drawer_slide').dofadr[0]
        self.cup_geoms = np.flatnonzero(
            (self.model.body_weldid[self.model.geom_bodyid] == self.cup_id)
            & ((self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)))
        robot_bodies = np.zeros(self.model.nbody, dtype=bool)
        robot_bodies[self.model.body('link0').id] = True
        for body in range(1, self.model.nbody):
            robot_bodies[body] |= robot_bodies[self.model.body_parentid[body]]
        self.robot_geoms = np.flatnonzero(robot_bodies[self.model.geom_bodyid])
        self.rng = np.random.default_rng(seed)
        self.reset(seed, orientation)

    def cup_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return world-axis bounds of the cup's box/cylinder collision geoms."""
        centers, extents = [], []
        for index in self.cup_geoms:
            rotation = self.data.geom_xmat[index].reshape(3, 3)
            size = self.model.geom_size[index]
            if self.model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX:
                extent = np.abs(rotation) @ size
            else:
                extent = size[0] * np.linalg.norm(rotation[:, :2], axis=1)
                extent += size[1] * np.abs(rotation[:, 2])
            centers.append(self.data.geom_xpos[index])
            extents.append(extent)
        return (np.min(np.array(centers) - extents, axis=0),
                np.max(np.array(centers) + extents, axis=0))

    def cup_contacts(self, geom_name:str) -> bool:
        """Return whether a cup collision geom contacts the named scene geom."""
        target = self.model.geom(geom_name).id
        return any((contact.geom1 == target and contact.geom2 in self.cup_geoms)
                   or (contact.geom2 == target and contact.geom1 in self.cup_geoms)
                   for contact in self.data.contact)

    def reset(self, seed:int|None=None, orientation:str='random') -> dict:
        """Return reset diagnostics after settling the requested pose on the counter."""
        if orientation not in ('random', *ORIENTATIONS):
            raise ValueError(f'Unknown cup orientation: {orientation}')
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        desired = str(self.rng.choice(ORIENTATIONS)) if orientation == 'random' else orientation
        for attempt in range(12):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:7] = HOME
            self.data.qpos[7:9] = .04
            self.data.ctrl[:] = np.r_[HOME, 255]
            self.data.qpos[self.drawer_qpos] = self.config.drawer_opening
            yaw = self.rng.uniform(-np.pi, np.pi)
            tilt = {'upright': 0, 'inverted': np.pi, 'sideways': np.pi / 2}[desired]
            yaw_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            tilt_quat = np.array([np.cos(tilt / 2), np.sin(tilt / 2), 0, 0])
            position = self.data.qpos[self.cup_qpos:self.cup_qpos + 7]
            position[:3] = [self.rng.uniform(.02, .24),
                            self.rng.uniform(.12, min(.36, self.config.counter_depth - .15)),
                            self.config.counter_height + .15]
            mujoco.mju_mulQuat(position[3:], yaw_quat, tilt_quat)
            mujoco.mj_forward(self.model, self.data)
            minimum, _ = self.cup_bounds()
            position[2] += self.config.counter_height - minimum[2] + .0003
            mujoco.mj_forward(self.model, self.data)

            stable_steps = 0
            for _ in range(int(4 / self.config.timestep)):
                mujoco.mj_step(self.model, self.data)
                if self.is_stable() and self.cup_contacts('countertop'):
                    stable_steps += 1
                else:
                    stable_steps = 0
                if stable_steps >= int(.2 / self.config.timestep):
                    break
            mujoco.mj_forward(self.model, self.data)
            if (stable_steps >= int(.2 / self.config.timestep)
                    and self.orientation() == desired and self.on_counter()):
                self.data.time = 0
                self.data.qfrc_applied[:] = 0
                self.initial_orientation = desired
                return self.status()
        raise RuntimeError(f'Could not settle a stable {desired} cup after 12 attempts.')

    def orientation(self) -> str:
        """Classify the cup's vertical axis from its current world rotation."""
        vertical = self.data.xmat[self.cup_id].reshape(3, 3)[2, 2]
        return 'upright' if vertical > .9 else 'inverted' if vertical < -.9 else 'sideways'

    def is_stable(self) -> bool:
        """Return whether cup linear/angular speeds are below reset tolerances."""
        velocity = self.data.qvel[self.cup_dof:self.cup_dof + 6]
        return bool(np.linalg.norm(velocity[:3]) < .003
                    and np.linalg.norm(velocity[3:]) < .025)

    def on_counter(self) -> bool:
        """Check countertop contact and the complete cup's footprint and height."""
        minimum, maximum = self.cup_bounds()
        return bool(self.cup_contacts('countertop')
                    and minimum[0] > -self.config.counter_width / 2 + .02
                    and maximum[0] < self.config.counter_width / 2 - .02
                    and minimum[1] > .02
                    and maximum[1] < self.config.counter_depth - .02
                    and abs(minimum[2] - self.config.counter_height) < .002)

    def step(self, control:np.ndarray|None=None, n_steps:int=10) -> None:
        """Advance physics with seven joint targets (radians) and gripper [0, 255]."""
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError('n_steps must be a positive integer.')
        if control is not None:
            control = np.asarray(control, dtype=float)
            limits = self.model.actuator_ctrlrange
            if (control.shape != (8,) or not np.isfinite(control).all()
                    or np.any(control < limits[:, 0]) or np.any(control > limits[:, 1])):
                raise ValueError('Expected 8 finite controls within actuator_ctrlrange.')
            self.data.ctrl[:] = control
        mujoco.mj_step(self.model, self.data, nstep=n_steps)
        mujoco.mj_forward(self.model, self.data)

    def observe(self, width:int=640, height:int=480, camera:str='rgbd') -> dict[str, np.ndarray]:
        """Return RGB uint8, metric depth float32, intrinsics, and camera pose.

        Depth is optical-axis distance in meters. camera_to_world uses optical
        axes: x right, y down, z forward; images have their origin at top left.
        Call mj_forward after editing raw state directly, as for other MuJoCo reads.
        """
        if not (0 < width <= 1280 and 0 < height <= 960):
            raise ValueError('Camera dimensions must be within 1280 by 960.')
        if self.renderer is None or (self.renderer.width, self.renderer.height) != (width, height):
            self.close()
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        camera = self.model.camera(camera).id
        focal = height / (2 * np.tan(np.deg2rad(self.model.cam_fovy[camera]) / 2))
        intrinsics = np.array([[focal, 0, (width - 1) / 2],
                               [0, focal, (height - 1) / 2], [0, 0, 1]])
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = self.data.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1])
        camera_to_world[:3, 3] = self.data.cam_xpos[camera]
        return dict(rgb=rgb, depth=depth, intrinsics=intrinsics,
                    camera_to_world=camera_to_world)

    def status(self) -> dict:
        """Return privileged diagnostics for validation, separate from RGBD input."""
        minimum, maximum = self.cup_bounds()
        center = self.data.body('drawer').xpos
        half = np.array([self.config.drawer_width, self.config.drawer_depth,
                         self.config.drawer_height]) / 2
        inside = bool(np.all(minimum - center >= -half + [.012, .012, .010])
                      and np.all(maximum - center <= half - [.012, .012, 0])
                      and self.cup_contacts('drawer_inner_bottom'))
        opening = float(self.data.qpos[self.drawer_qpos])
        robot_contact = any((c.geom1 in self.cup_geoms and c.geom2 in self.robot_geoms)
                            or (c.geom2 in self.cup_geoms and c.geom1 in self.robot_geoms)
                            for c in self.data.contact)
        return dict(cup_orientation=self.orientation(), cup_stable=self.is_stable(),
                    cup_on_counter=self.on_counter(), cup_in_drawer=inside,
                    cup_robot_contact=robot_contact, drawer_opening=opening,
                    success=inside and abs(opening) < .005 and not robot_contact,
                    cup_position=self.data.xpos[self.cup_id].tolist())

    def close(self) -> None:
        """Release the optional offscreen rendering context."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def run() -> None:
    """Run the viewer or a timed headless simulation and optionally export RGBD."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--orientation', choices=['random', *ORIENTATIONS], default='random')
    parser.add_argument('--randomize-scene', action='store_true')
    parser.add_argument('--handle', choices=['bar', 'knob', 'loop'])
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--drawer-demo', action='store_true',
                        help='Apply external forces to cycle the passive drawer (diagnostic).')
    parser.add_argument('--output', type=Path, help='Save RGB, depth, calibration, and scene.')
    parser.add_argument('--no-progress', action='store_true', help='Disable headless simulation progress.')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be finite and positive.')
    config = SceneConfig.randomized(args.seed) if args.randomize_scene else SceneConfig()
    if args.handle:
        config.handle = args.handle
    simulation = KitchenSimulation(config, args.seed, args.orientation)
    print(json.dumps(simulation.status(), indent=2))
    commands = []

    def advance():
        """Apply pending viewer commands and advance one simulation frame."""
        while commands:
            command = commands.pop(0)
            if command == ord('R'):
                simulation.reset(orientation=args.orientation)
            elif command == ord('G'):
                simulation.data.ctrl[7] = 0 if simulation.data.ctrl[7] > 127 else 255
            elif command == ord('D'):
                args.drawer_demo = not args.drawer_demo
        force = 0.0
        if args.drawer_demo:
            target = config.drawer_travel * .9 * (.5 - .5 * np.cos(2 * np.pi * simulation.data.time / 6))
            error = target - simulation.data.qpos[simulation.drawer_qpos]
            force = np.clip(160 * error - 25 * simulation.data.qvel[simulation.drawer_dof], -25, 25)
        simulation.data.qfrc_applied[simulation.drawer_dof] = force
        simulation.step()

    try:
        if args.headless:
            with progress_bar(total=args.seconds, desc='Simulating', unit='sim s',
                              disabled=args.no_progress) as steps:
                while simulation.data.time < args.seconds:
                    before = simulation.data.time
                    advance()
                    steps.update(min(simulation.data.time, args.seconds) - min(before, args.seconds))
        else:
            from mujoco import viewer as mj_viewer

            with mj_viewer.launch_passive(simulation.model, simulation.data,
                                          key_callback=commands.append) as viewer:
                viewer.cam.lookat[:] = [0, .12, config.counter_height - .15]
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                print('R: reset cup | G: open/close gripper | D: toggle drawer force demo')
                while viewer.is_running():
                    start = time.monotonic()
                    with viewer.lock():
                        advance()
                    viewer.sync()
                    time.sleep(max(0, 10 * config.timestep - (time.monotonic() - start)))
        if args.output:
            from PIL import Image

            observation = simulation.observe()
            args.output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(observation['rgb']).save(args.output / 'rgb.png')
            np.save(args.output / 'depth.npy', observation['depth'])
            np.savez_compressed(args.output / 'rgbd.npz', **observation)
            (args.output / 'scene.xml').write_text(simulation.xml)
            metadata = dict(seed=args.seed, initial_orientation=simulation.initial_orientation,
                            config=asdict(config), status=simulation.status(),
                            mujoco_version=mujoco.__version__, asset_sources=SOURCES)
            (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
            print(f'Saved RGBD and scene to {args.output.resolve()}')
        print(json.dumps(simulation.status(), indent=2))
    finally:
        simulation.close()


if __name__ == '__main__':
    run()
