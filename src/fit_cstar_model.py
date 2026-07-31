# -*- coding: utf-8 -*-
"""Fit descriptors -> c* on the clean set; LOO-CV + feature importance."""
import csv, re, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr

rows=list(csv.DictReader(open("cstar_dataset_clean.csv")))
FEATURES=["n_species","natoms","volume_per_atom","density","mean_electroneg","electroneg_spread",
          "mean_Z","max_Z","mean_row","valence_e_per_atom","grad_ratio","mp_pbe_gap","c_selfconsistent","pbe_gap",
          "band_center"]   # anion-p band center (eV, rel E_F) from a PBE DOS run (LORBIT=11)
def _num(v):
    m=re.match(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", str(v).strip())
    return float(m.group(0)) if m else float("nan")
X=np.array([[_num(r[f]) for f in FEATURES] for r in rows])
y=np.array([float(r["c_star"]) for r in rows])
slopes=np.array([float(r["dGap_dc"]) for r in rows])
n=len(y); print(f"n={n} materials, {len(FEATURES)} cheap descriptors")
print(f"c* : mean={y.mean():.3f} std={y.std():.3f} range=[{y.min():.2f},{y.max():.2f}]")
loo=LeaveOneOut()

def report(name, ypred):
    mae=np.mean(np.abs(ypred-y)); rmse=np.sqrt(np.mean((ypred-y)**2))
    ss=1-np.sum((y-ypred)**2)/np.sum((y-y.mean())**2)
    gap_mae=np.mean(np.abs(ypred-y)*slopes)  # implied fundamental-gap error (eV)
    print(f"  {name:22s} MAE(c*)={mae:.3f}  RMSE={rmse:.3f}  R2={ss:+.2f}  ~gapMAE={gap_mae:.2f} eV")
    return mae

print("\n=== LOO-CV baselines & models (predicting c*) ===")
report("baseline: mean c*", np.full(n,y.mean()))          # naive
report("baseline: c_selfconsist", X[:,FEATURES.index("c_selfconsistent")])  # raw physics c
report("Ridge(alpha=1)", cross_val_predict(make_pipeline(StandardScaler(),Ridge(alpha=1.0)),X,y,cv=loo))
report("RandomForest", cross_val_predict(RandomForestRegressor(n_estimators=400,random_state=0),X,y,cv=loo))
kern=ConstantKernel()*RBF(length_scale=np.ones(len(FEATURES)))+WhiteKernel()
report("GaussianProcess", cross_val_predict(make_pipeline(StandardScaler(),GaussianProcessRegressor(kernel=kern,normalize_y=True,n_restarts_optimizer=2,random_state=0)),X,y,cv=loo))

print("\n=== feature importance ===")
# 1) Pearson corr with c*
print("  Pearson r (feature vs c*):")
corr=sorted(((abs(pearsonr(X[:,i],y)[0]), FEATURES[i], pearsonr(X[:,i],y)[0]) for i in range(len(FEATURES))), reverse=True)
for a,f,r in corr: print(f"    {f:20s} r={r:+.3f}")
# 2) RF importances + permutation importance
rf=RandomForestRegressor(n_estimators=400,random_state=0).fit(X,y)
imp=sorted(zip(rf.feature_importances_,FEATURES),reverse=True)
print("\n  RandomForest impurity importance:")
for v,f in imp: print(f"    {f:20s} {v:.3f}")
pi=permutation_importance(rf,X,y,n_repeats=30,random_state=0)
pim=sorted(zip(pi.importances_mean,pi.importances_std,FEATURES),reverse=True)
print("\n  Permutation importance (mean±std):")
for m,s,f in pim: print(f"    {f:20s} {m:+.3f} ± {s:.3f}")
