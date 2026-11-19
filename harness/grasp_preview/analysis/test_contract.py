import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generate_candidates import previous
from preview import WaypointReader,DrawerSkills
from control import PreviewLibrary
import reseat

class ReadOnly:
    mode="full"
    diagnostic="current"
    def pos(self,name):return np.array([0.,0.,.5]) if name=="source_mug" else np.array([.3,0.,.6])
    def _body_frame(self,name):return np.eye(3)

class Contract(unittest.TestCase):
    def test_pointer_tracks_optional_native_grasps_without_stepping(self):
        reader=WaypointReader(ReadOnly())
        DrawerSkills._place(reader,"source_mug","a_table_left_region",0,1)
        self.assertEqual(len(reader.closes),6)
        np.testing.assert_allclose(reader.closes[0],[.028,.015,.585])
        np.testing.assert_allclose(reader.closes[-1],[.031,0.,.515])
    def test_budget_view_matches_actions_charged(self):
        lib=PreviewLibrary.__new__(PreviewLibrary)
        for used in range(3):
            lib.adjustments=[{}]*used
            self.assertEqual(lib.ledger(dict(corrections=99,retries_left=99)),
                dict(corrections=used,corrections_used=used,retries_left=2-used))
    def test_budget_covers_all_waypoints_and_opening(self):
        self.assertEqual(reseat.cost(dict(xy_gap=.06,end=[0,.06,0])),86)
        self.assertEqual(reseat.cost(dict(xy_gap=.2,end=[0,.2,0])),123)
if __name__=="__main__":unittest.main()
