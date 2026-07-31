#!/usr/bin/env python
"""Prepare data files for the 4 paper figures (MATLAB reads these). No plotting here."""
import csv, os, re, numpy as np
from sklearn.ensemble import RandomForestRegressor
from pymatgen.core import Composition, Element

OUT = "fig_data"; os.makedirs(OUT, exist_ok=True)
FEATURES = ["n_species","natoms","volume_per_atom","density","mean_electroneg","electroneg_spread",
            "mean_Z","max_Z","mean_row","valence_e_per_atom","grad_ratio","mp_pbe_gap",
            "c_selfconsistent","pbe_gap","band_center"]
def num(x):
    m=re.search(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?',str(x)); return float(m.group()) if m else np.nan

rows=[r for r in csv.DictReader(open("cstar_dataset.csv")) if r["qc_pass"]=="1"]
def anion(formula):
    try:
        els=[e for e in Composition(formula).elements]
        return max(els, key=lambda e: (e.X if e.X else 0)).symbol
    except Exception: return "?"
X=np.array([[num(r[f]) for f in FEATURES] for r in rows]); y=np.array([float(r["c_star"]) for r in rows])
Xno=np.array([[num(r[f]) for f in FEATURES if f!="band_center"] for r in rows])  # ablation: no band_center

def rf(): return RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
def loo(Xm):
    p=np.zeros(len(y))
    for i in range(len(y)):
        m=np.arange(len(y))!=i; g=rf(); g.fit(Xm[m],y[m]); p[i]=g.predict(Xm[i:i+1])[0]
    return p
pred=loo(X); pred_no=loo(Xno)
def r2(t,p): return 1-np.sum((t-p)**2)/np.sum((t-t.mean())**2)
print("LOO RF (15 feat): MAE=%.3f R2=%.3f" % (np.abs(pred-y).mean(), r2(y,pred)))

# ---- Fig 1 + Fig 2 + Fig 3-trends: master per-material table ----
with open(f"{OUT}/materials.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["formula","spacegroup","anion","ref_gap","pbe_gap","c_star",
                                 "electroneg_spread","band_center","regime","pred_cstar","is_chalco"])
    for i,r in enumerate(rows):
        w.writerow([r["formula"],r["spacegroup"],anion(r["formula"]),r["ref_gap"],r["mp_pbe_gap"],
                    r["c_star"],r["electroneg_spread"],r["band_center"],
                    "tail" if float(r["c_star"])>=1.6 else "dense",
                    round(pred[i],4), 1 if r["spacegroup"]=="I-42d" else 0])

# ---- Fig 2 fit stats ----
with open(f"{OUT}/parity_stats.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["mae","r2","n"]); w.writerow([round(np.abs(pred-y).mean(),4),round(r2(y,pred),4),len(y)])

# ---- Fig 3: RF feature importances (fit on all clean) ----
g=rf(); g.fit(X,y)
imp=sorted(zip(FEATURES,g.feature_importances_), key=lambda t:t[1])
with open(f"{OUT}/importance.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["feature","importance"])
    for k,v in imp: w.writerow([k,round(float(v),5)])

# ---- Fig 4: chalcopyrite band-center effect + ablation MAE ----
chi=[i for i,r in enumerate(rows) if r["spacegroup"]=="I-42d"]
mae_all_with=np.abs(pred-y).mean(); mae_all_no=np.abs(pred_no-y).mean()
mae_ch_with=np.abs(pred[chi]-y[chi]).mean(); mae_ch_no=np.abs(pred_no[chi]-y[chi]).mean()
non=[i for i in range(len(y)) if i not in chi]
mae_non_with=np.abs(pred[non]-y[non]).mean(); mae_non_no=np.abs(pred_no[non]-y[non]).mean()
print("chalco MAE: no-bc %.3f -> with-bc %.3f  (n=%d)"%(mae_ch_no,mae_ch_with,len(chi)))
with open(f"{OUT}/ablation.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["group","mae_no_bandcenter","mae_with_bandcenter"])
    w.writerow(["chalcopyrite",round(mae_ch_no,4),round(mae_ch_with,4)])
    w.writerow(["non_chalco",round(mae_non_no,4),round(mae_non_with,4)])
    w.writerow(["all",round(mae_all_no,4),round(mae_all_with,4)])
print("wrote %s/{materials,parity_stats,importance,ablation}.csv" % OUT)
