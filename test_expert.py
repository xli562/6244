"""Integration checks for real-contact task execution and isolated IK planning."""

import unittest

import mujoco
import numpy as np

from expert import Expert, ExpertFailure, apply_control
from simulation import KitchenSimulation, ORIENTATIONS


class ExpertTests(unittest.TestCase):
    def test_physical_task_all_orientations(self):
        """Complete the ordered task without applying forces or attaching objects."""
        sim = KitchenSimulation(seed=0)

        def inspect(simulation, action, stage):
            if simulation.data.qfrc_applied.any() or simulation.data.xfrc_applied.any():
                self.fail('Expert applied an external generalized or Cartesian force.')

        try:
            self.assertFalse(np.any(sim.model.eq_type == mujoco.mjtEq.mjEQ_WELD))
            for orientation in ORIENTATIONS:
                with self.subTest(orientation=orientation):
                    sim.reset(0, orientation)
                    report = Expert(sim, inspect).run()
                    self.assertTrue(report['success'], report)
                    self.assertTrue(all(report['progress'].values()))
                    self.assertTrue(sim.status()['cup_in_drawer'])
                    self.assertFalse(sim.status()['cup_robot_contact'])
                    self.assertTrue(sim.is_stable())
        finally:
            sim.close()

    def test_ik_does_not_change_live_physics(self):
        sim = KitchenSimulation(seed=0)
        try:
            positions, velocities = sim.data.qpos.copy(), sim.data.qvel.copy()
            expert = Expert(sim)
            with self.assertRaises(ExpertFailure):
                expert.solve_ik(np.array([10., 10, 10]), np.eye(3))
            np.testing.assert_array_equal(sim.data.qpos, positions)
            np.testing.assert_array_equal(sim.data.qvel, velocities)
        finally:
            sim.close()

    def test_observation_preserves_control_replay(self):
        """Reading RGBD during execution must not change the physics trajectory."""
        first, second = KitchenSimulation(seed=2), KitchenSimulation(seed=2)
        try:
            control = first.data.ctrl.copy()
            control[0] += .1
            for _ in range(3):
                first.observe(80, 60)
                apply_control(first, control)
                apply_control(second, control)
                np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
                np.testing.assert_array_equal(first.data.qvel, second.data.qvel)
        finally:
            first.close()
            second.close()

    def test_interpolation_at_actuator_limit(self):
        """Floating-point interpolation must stay inside the actuator's exact bounds."""
        sim = KitchenSimulation(seed=0)
        try:
            control = sim.data.ctrl.copy()
            control[0] = sim.model.actuator_ctrlrange[0, 1]
            sim.data.ctrl[:] = control
            apply_control(sim, control)
            self.assertTrue(np.all(sim.data.ctrl <= sim.model.actuator_ctrlrange[:, 1]))
        finally:
            sim.close()


if __name__ == '__main__':
    unittest.main()
