"""Checks for RGBD geometry, demonstrated waypoint transfer, and controller isolation."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from simulation import KitchenSimulation


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Install requirements-learning.txt.')
class WaypointTests(unittest.TestCase):
    def test_metric_depth_precision(self):
        """Depth reconstruction of a plane must remain accurate enough for physical grasps."""
        from waypoints import countertop_height, world_points

        sim = KitchenSimulation(seed=0)
        try:
            observation = sim.observe(384, 288, 'workspace_rgbd')
            points = world_points(observation)
            self.assertAlmostEqual(countertop_height(points), sim.config.counter_height, delta=.0001)
            camera = observation['camera_to_world']
            checked = 0
            for y in range(60, 180, 15):
                for x in range(100, 300, 15):
                    ray = np.linalg.solve(observation['intrinsics'], [x, y, 1])
                    ray /= np.linalg.norm(ray)
                    direction = camera[:3, :3] @ ray
                    geom = np.array([-1], np.int32)
                    distance = mujoco.mj_ray(sim.model, sim.data, camera[:3, 3], direction,
                                             np.array([1, 1, 1, 0, 0, 0], np.uint8), True, -1, geom)
                    hit = camera[:3, 3] + distance * direction
                    if (geom[0] == sim.model.geom('countertop').id
                            and abs(hit[2] - sim.config.counter_height) < 1e-6):
                        np.testing.assert_allclose(points[y, x], hit, atol=.0001)
                        checked += 1
            self.assertGreater(checked, 10)
        finally:
            sim.close()

    def test_demonstrated_waypoint_transfer(self):
        """Registration transfers expert targets under independently changed object poses."""
        import torch
        from waypoints import WaypointMemory

        torch.set_num_threads(2)
        rng = np.random.default_rng(42)
        points = rng.normal(0, .02, (1, 2, 128, 3)).astype(np.float32)
        points[:, 0, :, 2] = points[:, 0, :, 2].clip(-.015, .015)
        points += np.array([[[[.12, .25, .88]], [[.14, -.01, .65]]]], np.float32)
        target = np.array([[.13, .26, .87, .14, -.01, .65,
                            .005, -.01, .002, 1, 0, 0, 0, 1]], np.float32)
        memory = WaypointMemory(torch.from_numpy(points), torch.from_numpy(target))
        angle = .12
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
        cup_translation = np.array([.03, -.05, .02], np.float32)
        handle_translation = np.array([0, -.08, -.03], np.float32)
        transformed = np.stack([points[0, 0] @ rotation.T + cup_translation,
                                 points[0, 1] + handle_translation])
        xyz = torch.from_numpy(transformed.reshape(16, 16, 3).transpose(2, 0, 1)[None])
        dense = torch.full((1, 17, 16, 16), -10.)
        dense[:, 1, :8] = 10
        dense[:, 2, 8:] = 10
        predicted = memory.predict(dense, xyz)
        np.testing.assert_allclose(predicted[:3], rotation @ target[0, :3] + cup_translation, atol=1e-5)
        np.testing.assert_allclose(predicted[3:6], target[0, 3:6] + handle_translation, atol=1e-5)
        np.testing.assert_allclose(predicted[6:9], rotation @ target[0, 6:9], atol=1e-5)
        np.testing.assert_allclose(predicted[9:11], [np.cos(2 * angle), np.sin(2 * angle)], atol=1e-5)
        # A false-positive mask on the supporting plane must not move the grasp.
        table = torch.zeros(1, 3, 4, 16)
        table[:, 2] = .8
        extended_xyz = torch.cat([xyz, table], 2)
        false_positive = torch.full((1, 17, 4, 16), -10.)
        false_positive[:, 1] = 10
        filtered = memory.predict(torch.cat([dense, false_positive], 2), extended_xyz,
                                  support_height=.8)
        np.testing.assert_allclose(filtered, predicted, atol=1e-5)

    def test_comparison_requires_matched_conditions(self):
        """The research report rejects unequal training budgets or overlapping development seeds."""
        from waypoints import compare

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            paths = {name: directory / f'{name}.json' for name in ['none', 'cup', 'scene']}
            for count, (name, path) in enumerate(paths.items(), 1):
                report = dict(checkpoint=f'{name}.pt', decoder='registered', seed_overlap=[],
                              training_randomization=name, training_settings=dict(epochs=80, steps_per_epoch=40),
                              episodes=[dict(seed=10000 + index, orientation=orientation,
                                             config=dict(handle='bar'), success=index < count)
                                        for index, orientation in enumerate(['upright', 'inverted', 'sideways'])])
                path.write_text(json.dumps(report))
            args = argparse.Namespace(original_report=paths['none'], cup_report=paths['cup'],
                                      scene_report=paths['scene'], output=directory / 'summary.json')
            with redirect_stdout(io.StringIO()):
                compare(args)
            result = json.loads(args.output.read_text())
            self.assertEqual(result['paired_scene_vs_original'], dict(improved=2, regressed=0))
            altered = json.loads(paths['none'].read_text())
            altered['training_settings']['epochs'] = 81
            paths['none'].write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                compare(args)
            altered['training_settings']['epochs'] = 80
            altered['seed_overlap'] = [10000]
            paths['none'].write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                compare(args)

    def test_controller_cannot_query_object_state(self):
        """The waypoint executor completes a task while all oracle scene queries are forbidden."""
        from expert import ArmController
        from waypoints import execute, expert_labels

        sim = KitchenSimulation(seed=0, orientation='upright')
        try:
            sim.observe(384, 288, 'workspace_rgbd')
            _, target = expert_labels(sim)  # Labels are available only before the execution guard.
            with patch.object(sim, 'status', side_effect=AssertionError('Oracle status')), \
                    patch.object(sim, 'cup_bounds', side_effect=AssertionError('Oracle geometry')), \
                    patch.object(sim, 'on_counter', side_effect=AssertionError('Oracle contact')), \
                    patch('expert.Expert.run', side_effect=AssertionError('Expert fallback')):
                execute(ArmController(sim), target, .85)
            self.assertTrue(sim.status()['success'])
            self.assertFalse(sim.data.qfrc_applied.any())
            self.assertFalse(sim.data.xfrc_applied.any())
        finally:
            sim.close()


if __name__ == '__main__':
    unittest.main()
