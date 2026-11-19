"""Use the previous release study's independent native-predicate and state-hash audit."""
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import benchmark as bench
sys.path.insert(0,str(bench.ROOT/"harness/release_review/analysis"))
import attribution

def audit(stem):
    with patch.object(attribution,"runner",bench),patch.object(attribution,"ReviewLibrary",bench.PreviewLibrary):
        attribution.main(stem)
if __name__=="__main__":audit(sys.argv[1])
