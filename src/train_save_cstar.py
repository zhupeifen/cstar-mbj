# -*- coding: utf-8 -*-
"""Train the final c*-predictor on the clean set and save it (GP w/ uncertainty + RF backup)."""
import csv, re, json, numpy as np, joblib, datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict

FEATURES=["n_species","natoms","volume_per_atom","density","mean_electroneg","electroneg_spread",
          "mean_Z","max_Z","mean_row","valence_e_per_atom","grad_ratio","mp_pbe_gap","c_selfconsistent","pbe_gap",
          "band_center"]   # anion-p band center (eV, rel E_F) from a PBE DOS run (LORBIT=11)
def _num(v):
    m=re.match(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(v).strip()); return float(m.group(0)) if m else float("nan")
rows=list(csv.DictReader(open("cstar_dataset_clean.csv")))
X=np.array([[_num(r[f]) for f in FEATURES] for r in rows]); y=np.array([float(r["c_star"]) for r in rows])
loo=LeaveOneOut()

gp=Pipeline([("sc",StandardScaler()),
             ("gp",GaussianProcessRegressor(kernel=ConstantKernel()*RBF(np.ones(len(FEATURES)))+WhiteKernel(),
                                            normalize_y=True,n_restarts_optimizer=3,random_state=0))])
rf=RandomForestRegressor(n_estimators=400,random_state=0)
# CV metrics (for the record)
def cvm(m):
    p=cross_val_predict(m,X,y,cv=loo); mae=float(np.mean(np.abs(p-y)))
    r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)); return dict(loo_mae=round(mae,4),loo_r2=round(r2,3))
meta={"features":FEATURES,"n_train":len(y),"target":"c_star","cv_gp":cvm(gp),"cv_rf":cvm(rf),
      "trained":datetime.date.today().isoformat(),
      "note":"predict TB-mBJ optimal c* from cheap descriptors; GP gives sigma. pbe_gap==mp_pbe_gap (PBE gap)."}
gp.fit(X,y); rf.fit(X,y)
joblib.dump({"model":gp,"rf":rf,"meta":meta}, "cstar_model.joblib")
json.dump(meta, open("cstar_model.meta.json","w"), indent=2)
print("saved cstar_model.joblib +.meta.json"); print(json.dumps(meta,indent=2))
