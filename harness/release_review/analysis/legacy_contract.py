"""Apply only the new measurement contract on exactly the old archived error paths."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from unittest.mock import patch
import attribution as audit
from controller import ReviewLibrary
r=audit.runner

class NumericOnly(r.prior.StageLibrary):
    normalize=ReviewLibrary.normalize
    def configure(self,*args):
        super().configure(*args)
        self.observed=None
        self.peak=0.
        self.height_fixes=0
        api=self._sensors
        original=api.action
        def measure(*args,**kwargs):
            try:return original(*args,**kwargs)
            finally:
                if self.observed:
                    item,z=self.observed
                    self.peak=max(self.peak,float(api.pos(item)[2]-z))
        api.action=measure

    def scope(self):
        if self.observed or not self.current or self.current[0]["op"]!="place":return False
        item=self.current[1]["item"]
        self.observed=(item,self.pos(item)[2])
        self.peak=0.
        return True

    def perform(self,*args):
        own=self.scope()
        try:
            result=super().perform(*args)
            self.normalize()
            return dict(result,**self.raw)
        finally:
            if own:self.observed=None

    def recover(self,*args):
        own=self.scope()
        try:return super().recover(*args)
        finally:
            self.normalize()
            if own:self.observed=None

def one(x):
    k,t,i=audit.key(x)
    with patch.object(r.prior,"StageLibrary",NumericOnly):
        row=r.prior.play((k,x["arm"],t,[i]))[0]
    return dict(key=k,task=t,i=i,arm=x["arm"],old_error=x["error"],error=row["error"],stable=row["stable"])
if __name__=="__main__":
    jobs=[]
    for stem in ("test","challenge"):
        with gzip.open(r.prior.OUT/(stem+".jsonl.gz"),"rt") as f:
            jobs.extend(x for x in map(json.loads,f) if "NoneType" in str(x["error"]))
    with ProcessPoolExecutor(22) as pool,gzip.open(r.OUT/"legacy_contract.jsonl.gz","wt") as f:
        for row in pool.map(one,jobs):f.write(json.dumps(row)+"\n")
