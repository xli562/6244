import unittest
from generate import old  # add the frozen runtime to the module path
from recovery import revisit
import numpy as np

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.scene = {n:{"kind":"item","parent":None} for n in ("spool_1","spool_2","peg")}
        self.scene["desk_spool_left_mark"] = {"kind":"region","parent":"desk"}
        self.positions = {"spool_1":[0,0,0],"spool_2":[0.2,0,0],"peg":[0.19,0,0],"desk_spool_left_mark":[0.19,0,0]}
        self.event = dict(pick="spool_1",resolved=None,status="unresolved",finalists=["spool_1","spool_2"])
    def check(self, text, quote=None):
        return revisit(text,{"op":"place"},"item",self.event,self.scene,
                       lambda n:np.array(self.positions[n]),lambda *args:quote)
    def test_left(self):
        r=self.check("put the left spool on the peg","left spool")
        self.assertEqual((r["resolved"],r["revisit"]["via"]),("spool_2","agent"))
    def test_destination_word_cannot_leak(self):
        self.assertIsNone(self.check("put the spool on the left peg","left peg")["resolved"])
    def test_distance_tie(self):
        self.positions["desk_spool_left_mark"]=[0.1001,0,0]
        self.assertIsNone(self.check("put the left spool on the peg")["resolved"])
    def test_multiple_landmarks(self):
        self.scene["other_spool_left_mark"] = self.scene["desk_spool_left_mark"]
        self.assertIsNone(self.check("put the left spool on the peg")["resolved"])
    def test_named_reference(self):
        self.assertEqual(self.check("pick up the spool beside the peg and put it on the desk")["resolved"],"spool_2")
    def test_complex_scope(self):
        self.assertIsNone(self.check("put the spool on the left tray in the case")["resolved"])
    def test_multiple_steps_without_quote(self):
        self.assertIsNone(self.check("put the left spool on the peg and put the spool on the desk")["resolved"])
    def test_role_whose_relation_is_to_another_entity(self):
        self.assertIsNone(self.check("put the spool to the left of the peg on the desk")["resolved"])

if __name__=="__main__":
    unittest.main()
