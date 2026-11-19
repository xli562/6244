"""Regression cases for wrong nominations, ambiguous poses, and moving anchors."""

import unittest

import numpy as np

from binding import check_role


class BindingTest(unittest.TestCase):
    def setUp(self):
        self.scene = {
            **{f"token_{n}": {"kind": "item", "parent": None} for n in range(3)},
            "table_mark": {"kind": "region", "parent": "table"},
            "a_slot": {"kind": "region", "parent": "a"},
            "b_slot": {"kind": "region", "parent": "b"},
        }
        self.xyz = {
            n: np.array([v, 0.0, 0.0])
            for n, v in zip(self.scene, (2.0, 0.0, 1.0, 1.01, 0.1, 0.1))
        }

    def check(self, pick, **spec):
        return check_role(pick, spec, self.scene, self.xyz.__getitem__)

    def test_arbitrary_instance_numbers_and_live_pose(self):
        role = {"kind": "item", "terms": ["token"], "near": "table_mark"}
        self.assertEqual(self.check("token_0", **role)["resolved"], "token_2")
        self.xyz["table_mark"][0] = 0.02
        self.assertEqual(self.check("token_0", **role)["resolved"], "token_1")

    def test_median_and_near_tie(self):
        self.assertEqual(
            self.check(
                "token_1", kind="item", terms=["token"], axis="x", rank="median"
            )["resolved"],
            "token_2",
        )
        self.xyz["table_mark"][0] = 0.5005
        self.assertIsNone(
            self.check("token_1", kind="item", terms=["token"], near="table_mark")[
                "resolved"
            ]
        )

    def test_destination_owner_and_unknown(self):
        self.assertEqual(
            self.check("b_slot", kind="region", terms=["slot"], parent="a")["resolved"],
            "a_slot",
        )
        self.assertIsNone(
            self.check("a_slot", kind="region", terms=["slot"], parent="absent")[
                "resolved"
            ]
        )


if __name__ == "__main__":
    unittest.main()
