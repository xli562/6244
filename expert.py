"""Contact-based expert demonstrations for cup placement; uses simulator state."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from progress import add_progress_arguments, progress_bar
from simulation import KitchenSimulation, ORIENTATIONS, SceneConfig


CONTROL_DT = 0.1
FRONT = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])
DOWN = np.diag([1., -1, -1])


def proprioception(sim:KitchenSimulation) -> np.ndarray:
    """Return robot joint positions, velocities, and previous actuator targets."""
    return np.r_[sim.data.qpos[:9], sim.data.qvel[:9], sim.data.ctrl].astype(np.float32)


def apply_control(sim:KitchenSimulation, control:np.ndarray) -> None:
    """Interpolate one 10 Hz actuator target through five 50 Hz physics updates."""
    previous = sim.data.ctrl.copy()
    lower, upper = sim.model.actuator_ctrlrange.T
    for fraction in np.linspace(.2, 1, 5):
        command = np.r_[(1 - fraction) * previous[:7] + fraction * control[:7], control[7]]
        sim.step(np.clip(command, lower, upper),
                 n_steps=round(CONTROL_DT / 5 / sim.config.timestep))


class ExpertFailure(RuntimeError):
    pass


class ArmController:
    def __init__(self, sim:KitchenSimulation, callback:Callable|None=None):
        """Control the robot; callback receives (sim, target, stage) before control."""
        self.sim = sim
        self.callback = callback
        self.kinematics = mujoco.MjData(sim.model)
        self.site = sim.model.site('grasp').id
        self.joints = sim.data.qpos[:7].copy()
        self.stages = []
        self.stage = 'start'

    def solve_ik(self, position:np.ndarray, rotation:np.ndarray) -> np.ndarray:
        """Return bounded arm angles for a target world pose using damped Jacobian IK."""
        model, data = self.sim.model, self.kinematics
        data.qpos[:9] = self.sim.data.qpos[:9]
        data.qpos[:7] = self.joints
        jac_position = np.zeros((3, model.nv))
        jac_rotation = np.zeros((3, model.nv))
        for _ in range(100):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            current = data.site_xmat[self.site].reshape(3, 3)
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, (rotation @ current.T).ravel())
            if quaternion[0] < 0:
                quaternion *= -1
            angular_error = np.empty(3)
            mujoco.mju_quat2Vel(angular_error, quaternion, 1)
            error = np.r_[position - data.site_xpos[self.site], .3 * angular_error]
            if np.linalg.norm(error) < .0003:
                self.joints = data.qpos[:7].copy()
                return self.joints
            mujoco.mj_jacSite(model, data, jac_position, jac_rotation, self.site)
            jacobian = np.vstack([jac_position[:, :7], .3 * jac_rotation[:, :7]])
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .0001 * np.eye(6), error)
            delta *= min(1, .15 / (np.max(np.abs(delta)) + 1e-9))
            data.qpos[:7] = np.clip(data.qpos[:7] + delta,
                                    model.jnt_range[:7, 0] + .001,
                                    model.jnt_range[:7, 1] - .001)
        raise ExpertFailure(f'Unreachable pose during {self.stage}: {position.tolist()}')

    def advance(self, control:np.ndarray) -> None:
        """Apply a robot target and report numerical physics warnings."""
        if self.callback:
            self.callback(self.sim, control.copy(), self.stage)
        apply_control(self.sim, control)
        if self.sim.data.warning.number.any():
            raise ExpertFailure(f'Physics warning during {self.stage}')

    def move(self, position, rotation:np.ndarray, gripper:float=255,
             seconds:float=2, stage:str='move') -> None:
        """Move the grasp frame smoothly to a world pose using robot actuator commands."""
        self.stage = stage
        self.stages.append(stage)
        sim = self.sim
        start = sim.data.site('grasp').xpos.copy()
        start_quaternion, end_quaternion = np.empty(4), np.empty(4)
        mujoco.mju_mat2Quat(start_quaternion, sim.data.site('grasp').xmat)
        mujoco.mju_mat2Quat(end_quaternion, rotation.ravel())
        if start_quaternion @ end_quaternion < 0:
            end_quaternion *= -1
        position = np.asarray(position)
        self.joints = sim.data.qpos[:7].copy()
        for fraction in np.linspace(0, 1, round(seconds / CONTROL_DT) + 1)[1:]:
            smooth = fraction ** 2 * (3 - 2 * fraction)
            quaternion = (1 - smooth) * start_quaternion + smooth * end_quaternion
            quaternion /= np.linalg.norm(quaternion)
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, quaternion)
            joints = self.solve_ik((1 - smooth) * start + smooth * position, matrix.reshape(3, 3))
            # Compensate gravity through the existing position actuators, rather than object forces.
            command = joints + sim.data.qfrc_bias[:7] / sim.model.actuator_gainprm[:7, 0]
            command = np.clip(command, sim.model.actuator_ctrlrange[:7, 0],
                               sim.model.actuator_ctrlrange[:7, 1])
            self.advance(np.r_[command, gripper])
        self.hold(.2, gripper)
        error = np.linalg.norm(position - sim.data.site('grasp').xpos)
        if error > .025:
            raise ExpertFailure(f'Pose tracking error {error:.3f} m during {stage}')

    def hold(self, seconds:float, gripper:float) -> None:
        """Hold arm position for the requested time while commanding the gripper."""
        for _ in range(round(seconds / CONTROL_DT)):
            self.advance(np.r_[self.sim.data.ctrl[:7], gripper])


class Expert(ArmController):
    def __init__(self, sim:KitchenSimulation, callback:Callable|None=None):
        super().__init__(sim, callback)
        self.progress = dict(drawer_opened=False, cup_lifted=False, cup_placed=False,
                             drawer_closed=False)

    def advance(self, control:np.ndarray) -> None:
        """Apply an expert action and track privileged task metrics for demonstrations."""
        super().advance(control)
        status = self.sim.status()
        self.progress['drawer_opened'] |= status['drawer_opening'] > .28
        minimum, _ = self.sim.cup_bounds()
        self.progress['cup_lifted'] |= bool(self.progress['drawer_opened']
                                          and minimum[2] > self.sim.config.counter_height + .06)
        self.progress['cup_placed'] |= self.progress['cup_lifted'] and status['cup_in_drawer']
        self.progress['drawer_closed'] |= self.progress['cup_placed'] and status['success']

    def handle_pose(self) -> np.ndarray:
        """Return the actual grasp point for each drawer-handle shape."""
        name = {'bar': 'handle_bar', 'knob': 'handle_knob',
                'loop': 'handle_loop_middle'}[self.sim.config.handle]
        return self.sim.data.geom(name).xpos.copy()

    def run(self) -> dict:
        """Execute opening, pickup, placement, and closing; return honest stage diagnostics."""
        sim = self.sim
        try:
            handle = self.handle_pose()
            closed_handle = handle + [0, sim.data.qpos[sim.drawer_qpos], 0]
            self.move(handle + [0, -.18, .23], FRONT, seconds=3, stage='approach_drawer')
            self.move(handle + [0, -.14, 0], FRONT, stage='lower_to_handle')
            self.move(handle, FRONT, stage='reach_handle')
            self.stage = 'grasp_handle'
            self.hold(.8, 0)
            self.move(closed_handle + [0, -.34, 0], FRONT, 0, 4, 'open_drawer')
            if sim.data.qpos[sim.drawer_qpos] < .28:
                raise ExpertFailure('Handle grasp slipped while opening the drawer')
            self.hold(.5, 255)
            self.move(sim.data.site('grasp').xpos + [0, -.04, 0], FRONT, stage='release_handle')
            self.move(sim.data.site('grasp').xpos + [0, 0, .40], DOWN, stage='clear_drawer')

            cup_rotation = sim.data.body('cup').xmat.reshape(3, 3).copy()
            grasp = sim.data.body('cup').xpos + cup_rotation @ [0, 0, .037 * sim.config.cup_scale]
            axis = cup_rotation[:, 1].copy()
            axis[2] = 0
            if np.linalg.norm(axis) < .5:
                axis = cup_rotation[:, 0].copy()
                axis[2] = 0
                grasp = sim.data.body('cup').xpos + cup_rotation @ [0, 0, .0026 * sim.config.cup_scale]
            axis /= np.linalg.norm(axis)
            if axis[1] > 0:
                axis *= -1
            down = np.column_stack([np.cross(axis, [0, 0, -1]), axis, [0, 0, -1]])
            self.move(np.r_[grasp[:2], sim.config.counter_height + .25], down,
                      seconds=3, stage='approach_cup')
            self.move(grasp, down, stage='reach_cup')
            self.stage = 'grasp_cup'
            self.hold(.8, 0)
            self.move(grasp + [0, 0, .22], down, 0, 2, 'lift_cup')
            if sim.on_counter() or not self.progress['cup_lifted']:
                raise ExpertFailure('Cup grasp failed during lift')

            open_handle = self.handle_pose()
            minimum, maximum = sim.cup_bounds()
            offset = (minimum + maximum) / 2 - sim.data.site('grasp').xpos
            place = open_handle + [0, .20, .02] - offset
            self.move(np.r_[place[:2], sim.config.counter_height + .25], down,
                      0, 3, 'carry_cup')
            self.move(place, down, 0, 3, 'lower_cup')
            self.stage = 'release_cup'
            self.hold(.8, 255)
            exit_direction = sim.data.body('cup').xmat.reshape(3, 3)[:, 2].copy()
            if not sim.status()['cup_in_drawer'] and abs(exit_direction[2]) < .5:
                # A finger can enter a sideways cup; withdraw through its open rim.
                exit_direction[2] = 0
                exit_direction /= np.linalg.norm(exit_direction)
                place += .09 * exit_direction
                self.move(place, down, seconds=1.5, stage='withdraw_from_cup')
            self.move(place + [0, 0, .30], down, stage='retreat_from_cup')
            if not sim.status()['cup_in_drawer']:
                raise ExpertFailure('Cup was not contained after release')

            open_handle = self.handle_pose()
            self.move(open_handle + [0, -.08, .22], FRONT, seconds=3, stage='approach_closing')
            self.move(open_handle + [0, -.06, 0], FRONT, stage='lower_for_closing')
            self.move(open_handle, FRONT, stage='reach_for_closing')
            self.stage = 'grasp_for_closing'
            self.hold(.8, 0)
            self.move(closed_handle, FRONT, 0, 4, 'close_drawer')
            self.hold(.5, 255)
            self.move(closed_handle + [0, -.13, 0], FRONT, stage='release_drawer')
            self.hold(.5, 255)
            success = bool(all(self.progress.values()) and sim.status()['success'] and sim.is_stable())
            failure = None if success else 'Final containment/closure check failed'
        except ExpertFailure as error:
            success, failure = False, str(error)
        return dict(success=success, failure=failure, stage=self.stage,
                    progress=self.progress, seconds=float(sim.data.time), status=sim.status())


def run() -> None:
    """Execute one expert trial and optionally write a GIF and evaluation report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--orientation', choices=['random', *ORIENTATIONS], default='random')
    parser.add_argument('--randomize-scene', action='store_true')
    parser.add_argument('--handle', choices=['bar', 'knob', 'loop'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--video', action='store_true')
    add_progress_arguments(parser)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive.')
    reports = []
    trials = progress_bar(range(args.episodes), desc='Expert', unit='trial', disabled=args.no_progress)
    for index in trials:
        seed = args.seed + index
        config = SceneConfig.randomized(seed) if args.randomize_scene else SceneConfig()
        if args.handle:
            config.handle = args.handle
        orientation = (ORIENTATIONS[index % 3] if args.episodes > 1 and args.orientation == 'random'
                       else args.orientation)
        sim = KitchenSimulation(config, seed, orientation)
        frames, count, last_stage = [], 0, None

        def capture(simulation, control, stage):
            """Record a 2 Hz preview from the same fixed camera used for observations."""
            nonlocal count, last_stage
            if not trials.disable and stage != last_stage:
                trials.set_postfix_str(f'{seed} {stage.replace("_", " ")}')
                last_stage = stage
            if args.output and args.video and count % 5 == 0:
                from PIL import Image
                frames.append(Image.fromarray(simulation.observe(320, 240)['rgb']))
            count += 1

        try:
            report = Expert(sim, capture).run()
            report.update(seed=seed, orientation=sim.initial_orientation,
                          config=asdict(config), mujoco_version=mujoco.__version__)
            if args.output:
                output = args.output if args.episodes == 1 else args.output / f'seed{seed}'
                output.mkdir(parents=True, exist_ok=True)
                (output / 'report.json').write_text(json.dumps(report, indent=2))
                if frames:
                    frames[0].save(output / 'rollout.gif', save_all=True,
                                   append_images=frames[1:], duration=100, loop=0)
            reports.append(report)
            trials.set_postfix(ok=f"{sum(item['success'] for item in reports)}/{len(reports)}", refresh=False)
            if args.verbose:
                trials.write(json.dumps(report))
        finally:
            sim.close()
    summary = dict(policy='state_based_expert', episodes=reports,
                   successes=sum(report['success'] for report in reports), trials=len(reports))
    if args.output and args.episodes > 1:
        (args.output / 'benchmark.json').write_text(json.dumps(summary, indent=2))
    print(f"Expert success: {summary['successes']}/{summary['trials']}")
    if args.output:
        print(f'Reports saved to {args.output}')


if __name__ == '__main__':
    run()
