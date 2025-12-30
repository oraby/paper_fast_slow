import pandas as pd
from pathlib import Path
import pickle
try:
  import dill
except:
  pass
from typing import Union, Callable

def loadCachedOrRunNSave(lambdaFunc : Callable, fp : Union[Path,str],
                         dscrp : str=None):
  '''Example:
  from .. cache import loadCachedOrRunNSave as ld
  res = ld(lambda:runLongAnalysis(), "./long_analysis.pkl" ,"Long analysis")
  '''
  if not isinstance(fp, Path):
    fp = Path(fp)
  if dscrp: dscrp = f"{dscrp} "
  if fp.exists():
    print(f"Loading existing {dscrp}file for {fp.parent.name}/{fp.name}")
    # res = pd.read_pickle(fp)
    with open(fp, 'rb') as f:
      if fp.suffix == '.dill':
        res = dill.load(f)
      else:
        res = pickle.load(f)
  else:
    print(f"File not found. Processing for {fp} {dscrp}")
    res = lambdaFunc()
    if res is not None:
      print(f"Saving {fp} {dscrp}results to desk")
      if fp.suffix == '.dill':
        with open(fp, 'wb') as f:
          dill.dump(res, f)
      else:
        if isinstance(res, pd.DataFrame):
          res.to_pickle(fp)
        else:
          with open(fp, 'wb') as f:
            pickle.dump(res, f)
  return res
