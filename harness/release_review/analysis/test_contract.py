"""Regression tests for the measured-height contract and repeated grasp warnings."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller import ReviewLibrary

class ContractTest(unittest.TestCase):
    def test_interrupted_placement_reports_measured_height(self):
        for peak,reported,expected in [(0,None,0),(.034,None,.034),(.012,.025,.025)]:
            lib=ReviewLibrary.__new__(ReviewLibrary)
            lib.raw=dict(op="place",xy_gap=.08,max_lift=reported)
            lib.peak=peak
            lib.height_fixes=0
            lib.normalize()
            self.assertEqual(lib.raw["max_lift"],expected)
            self.assertEqual(lib.height_fixes,int(reported is None))

    def test_preserve_local_search_until_the_evidence_changes(self):
        lib=ReviewLibrary.__new__(ReviewLibrary)
        lib.mode="new"
        lib.approach={"wide":False}
        lib.closures=1
        lib.adjustments=[]
        lib.decisions=[]
        lib.continuation=None
        calls=[]
        def decide(event):
            calls.append(event)
            return "switch" if event["axis_change"] else "continue"
        lib.policy=decide
        lib.parent_handler=lambda _: "regrasp"
        event=dict(reason="empty_grasp",phase="grasp",contacts=0,object_motion=.00004,axis_change=.00001)
        self.assertIsNone(lib.arbitrate(event))
        self.assertEqual(calls[0]["axis_change"],0)
        lib.closures=2
        self.assertIsNone(lib.arbitrate(event))
        self.assertEqual(len(calls),1)
        self.assertEqual(lib.arbitrate(dict(event,axis_change=.4)),"regrasp")
        self.assertEqual(len(calls),2)

if __name__=="__main__":unittest.main()
