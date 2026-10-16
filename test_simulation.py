"""Physics and camera checks; run with MUJOCO_GL=egl python -m unittest -v."""

import unittest

import mujoco
import numpy as np

from simulation import KitchenSimulation, ORIENTATIONS, SceneConfig


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulation = KitchenSimulation(seed=0)

    @classmethod
    def tearDownClass(cls):
        cls.simulation.close()

    def test_stable_counter_resets(self):
        """Check seeded starts retain each orientation and real support contact."""
        sim = self.simulation
        for orientation in ORIENTATIONS:
            for seed in range(5):
                with self.subTest(orientation=orientation, seed=seed):
                    status = sim.reset(seed, orientation)
                    self.assertEqual(status['cup_orientation'], orientation)
                    self.assertTrue(status['cup_stable'])
                    self.assertTrue(status['cup_on_counter'])
                    self.assertEqual(sim.data.time, 0)
                    sim.step(n_steps=500)
                    self.assertTrue(sim.on_counter())
                    self.assertTrue(sim.is_stable())
                    self.assertLess(max(-c.dist for c in sim.data.contact), .001)
                    self.assertFalse(sim.data.warning.number.any())

    def test_reset_reproducibility_and_diversity(self):
        """Identical seeds reproduce states; distinct seeds vary pose and position."""
        sim = self.simulation
        sim.reset(42)
        expected = sim.data.qpos.copy()
        sim.step(n_steps=300)
        sim.reset(42)
        np.testing.assert_array_equal(sim.data.qpos, expected)
        states, orientations = [], set()
        for seed in range(12):
            status = sim.reset(seed)
            states.append(status['cup_position'][:2])
            orientations.add(status['cup_orientation'])
        self.assertEqual(orientations, set(ORIENTATIONS))
        self.assertGreater(np.ptp(np.array(states)[:, 0]), .1)
        self.assertGreater(np.ptp(np.array(states)[:, 1]), .1)

    def test_passive_drawer_travel(self):
        """External forces open/close the unactuated drawer without deep overlap."""
        sim = self.simulation
        sim.reset(0)
        self.assertEqual(sim.model.nu, 8)
        sim.data.qfrc_applied[sim.drawer_dof] = 12
        sim.step(n_steps=750)
        opening = sim.data.qpos[sim.drawer_qpos]
        self.assertAlmostEqual(opening, sim.config.drawer_travel, delta=.002)
        self.assertLess(max(-c.dist for c in sim.data.contact), .001)
        sim.data.qfrc_applied[sim.drawer_dof] = -12
        sim.step(n_steps=750)
        self.assertAlmostEqual(sim.data.qpos[sim.drawer_qpos], 0, delta=.002)
        self.assertTrue(sim.on_counter())
        self.assertFalse(sim.data.warning.number.any())

    def test_cup_support_in_drawer(self):
        """A diagnostic placement settles inside, stays contained, and closes."""
        sim = self.simulation
        sim.reset(3, 'upright')
        sim.data.qpos[sim.drawer_qpos] = .30
        mujoco.mj_forward(sim.model, sim.data)
        sim.data.qpos[sim.cup_qpos:sim.cup_qpos + 3] = (
            sim.data.body('drawer').xpos
            + [0, -.10, -sim.config.drawer_height / 2 + .014])
        sim.data.qvel[:] = 0
        mujoco.mj_forward(sim.model, sim.data)
        sim.step(n_steps=500)
        self.assertTrue(sim.status()['cup_in_drawer'])
        self.assertFalse(sim.status()['success'])
        for _ in range(1000):
            sim.data.qfrc_applied[sim.drawer_dof] = (
                -30 * sim.data.qpos[sim.drawer_qpos]
                - 15 * sim.data.qvel[sim.drawer_dof] - 2)
            sim.step(n_steps=1)
        self.assertTrue(sim.status()['success'])
        self.assertFalse(sim.status()['cup_on_counter'])
        self.assertFalse(sim.data.warning.number.any())

    def test_rgbd_calibration(self):
        """Check aligned camera arrays and compare metric depth to independent rays."""
        sim = self.simulation
        sim.reset(1, 'upright')
        observation = sim.observe(width=320, height=240)
        rgb, depth = observation['rgb'], observation['depth']
        self.assertEqual(rgb.shape, (240, 320, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(depth.shape, (240, 320))
        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(np.isfinite(depth).all())
        self.assertTrue((depth > 0).all())
        self.assertGreater(float(rgb.std()), 10)
        intrinsic = observation['intrinsics']
        pose = observation['camera_to_world']
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-10)
        for x, y in [(210, 45), (190, 150), (100, 220)]:
            ray = np.linalg.solve(intrinsic, [x, y, 1])
            distance_scale = np.linalg.norm(ray)
            direction = pose[:3, :3] @ (ray / distance_scale)
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(sim.model, sim.data, pose[:3, 3], direction,
                                     np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8),
                                     True, -1, geom_id)
            self.assertGreater(distance, 0)
            self.assertAlmostEqual(depth[y, x], distance / distance_scale, delta=.02)
        # Reuse and then resize the renderer to exercise context ownership.
        np.testing.assert_array_equal(sim.observe(320, 240)['depth'], depth)
        self.assertEqual(sim.observe(160, 120)['rgb'].shape, (120, 160, 3))
        sim.close()

    def test_handle_and_geometry_variants(self):
        """Different heights, drawer sizes, and handle types preserve stable physics."""
        for seed, handle in enumerate(['bar', 'knob', 'loop']):
            config = SceneConfig.randomized(seed)
            config.handle = handle
            with self.subTest(handle=handle):
                sim = KitchenSimulation(config, seed)
                try:
                    self.assertTrue(sim.on_counter())
                    sim.data.qfrc_applied[sim.drawer_dof] = 12
                    sim.step(n_steps=750)
                    self.assertAlmostEqual(sim.data.qpos[sim.drawer_qpos],
                                           config.drawer_travel, delta=.002)
                    self.assertFalse(sim.data.warning.number.any())
                finally:
                    sim.close()
        config = SceneConfig(drawer_opening=.2)
        sim = KitchenSimulation(config, seed=0)
        self.assertAlmostEqual(sim.status()['drawer_opening'], .2, delta=.001)
        sim.close()

    def test_robot_and_gripper_controls(self):
        """Joint position targets and the real finger mechanism respond to controls."""
        sim = self.simulation
        sim.reset(3)
        initial = sim.data.qpos[0]
        control = sim.data.ctrl.copy()
        control[0] += .15
        control[7] = 0
        sim.step(control, n_steps=750)
        self.assertGreater(sim.data.qpos[0] - initial, .10)
        self.assertLess(sim.data.qpos[7], .005)
        control[7] = 255
        sim.step(control, n_steps=500)
        self.assertGreater(sim.data.qpos[7], .035)
        self.assertFalse(sim.data.warning.number.any())

    def test_invalid_inputs(self):
        sim = self.simulation
        for control in [np.zeros(7), np.full(8, np.nan), np.full(8, 1000)]:
            with self.assertRaises(ValueError):
                sim.step(control)
        with self.assertRaises(ValueError):
            sim.reset(orientation='floating')
        with self.assertRaises(ValueError):
            sim.step(n_steps=0)
        with self.assertRaises(ValueError):
            sim.observe(width=2000)
        for kwargs in [dict(drawer_depth=.8), dict(handle='unknown'),
                       dict(counter_height=float('nan')), dict(drawer_clearance=.06)]:
            with self.assertRaises(ValueError):
                SceneConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
