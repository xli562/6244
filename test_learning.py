"""Checks for causal observations, dataset isolation, splits, and policy execution."""

import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Install requirements-learning.txt.')
class LearningTests(unittest.TestCase):
    def test_causal_history_and_privileged_data_exclusion(self):
        """Future robot states and diagnostic labels cannot alter a policy observation."""
        from learning import Demonstrations

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.npz'
            arrays = dict(rgb=np.zeros((5, 16, 24, 3), dtype=np.uint8),
                          depth=np.ones((5, 16, 24), dtype=np.float16),
                          proprioception=np.arange(5 * 26, dtype=np.float32).reshape(5, 26) / 100,
                          actions=np.zeros((5, 8), dtype=np.float32),
                          stage=np.array(['privileged'] * 5),
                          cup_pose=np.zeros((5, 7), dtype=np.float32))
            np.savez(path, **arrays)
            original = Demonstrations([path], 3)[2]
            arrays['proprioception'][3:] = 100000
            arrays['cup_pose'][:] = 100000
            arrays['stage'][:] = 'other'
            np.savez(path, **arrays)
            changed = Demonstrations([path], 3)[2]
            for expected, actual in zip(original, changed):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(original[1].shape, (10, 26))

    def test_configuration_split(self):
        from learning import split_episodes

        entries = [dict(seed=seed, orientation='upright', config={'height': .85}, file=str(index))
                   for index, seed in enumerate([0, 0, 1, 2, 2, 3])]
        training, validation = split_episodes(entries, 42)
        self.assertFalse({entry['seed'] for entry in training}
                         & {entry['seed'] for entry in validation})
        self.assertEqual(len(training) + len(validation), len(entries))
        training, validation = split_episodes(entries[:2], 42)
        self.assertEqual(len(training), 2)
        self.assertEqual(validation, [])

    def test_policy_visual_and_robot_gradients(self):
        """RGB, depth, and robot history all contribute to the learned action output."""
        import torch
        from learning import Policy

        torch.manual_seed(0)
        torch.set_num_threads(2)
        policy = Policy([16, 24], 4, np.zeros(26), np.ones(26))
        image = torch.rand(2, 4, 16, 24, requires_grad=True)
        robot = torch.rand(2, 10, 26, requires_grad=True)
        output = policy(image, robot)
        self.assertEqual(output.shape, (2, 4, 8))
        output.square().mean().backward()
        self.assertGreater(float(image.grad[:, :3].abs().sum()), 0)
        self.assertGreater(float(image.grad[:, 3].abs().sum()), 0)
        self.assertGreater(float(robot.grad.abs().sum()), 0)

    def test_evaluation_has_no_expert_fallback(self):
        import torch
        from learning import Policy, evaluate

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'policy.pt'
            output = Path(directory) / 'evaluation.json'
            policy = Policy([16, 24], 4, np.zeros(26), np.ones(26))
            torch.save(dict(state=policy.state_dict(), image_shape=[16, 24], horizon=4,
                            mean=[0.] * 26, std=[1.] * 26, history_length=10,
                            training_episodes=[], validation_episodes=[]), checkpoint)
            args = argparse.Namespace(checkpoint=checkpoint, output=output, device='cpu',
                                      seed_start=1000, episodes=1, randomization='cup',
                                      allow_training_seeds=False, seconds=.2, execute=1)
            oracle_guard = patch('expert.Expert.run', side_effect=AssertionError('Oracle fallback'))
            with oracle_guard, redirect_stdout(io.StringIO()):
                evaluate(args)
            report = json.loads(output.read_text())
            self.assertEqual(report['trials'], 1)
            self.assertEqual(report['seed_overlap'], [])
            self.assertFalse(report['episodes'][0]['success'])
            args.seed_start = 42
            saved = torch.load(checkpoint, weights_only=True)
            saved['training_episodes'] = [dict(seed=42)]
            torch.save(saved, checkpoint)
            with self.assertRaises(ValueError):
                evaluate(args)


if __name__ == '__main__':
    unittest.main()
