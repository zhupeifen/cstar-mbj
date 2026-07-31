#!/usr/bin/env python
"""Generate manuscript Tables I-III as markdown (pipe tables) -> fig_data/tables.md"""
import csv, re, json, numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
def num(x):
    m=re.search(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?',str(x)); return float(m.group()) if m else np.nan
FEATURES=["n_species","natoms","volume_per_atom","density","mean_electroneg","electroneg_spread",
          "mean_Z","max_Z","mean_row","valence_e_per_atom","grad_ratio","mp_pbe_gap",
          "c_selfconsistent","pbe_gap","band_center"]
rows=list(csv.DictReader(open("cstar_dataset.csv")))
clean=[r for r in rows if r["qc_pass"]=="1"]
X=np.array([[num(r[f]) for f in FEATURES] for r in clean]); y=np.array([float(r["c_star"]) for r in clean])
def r2(t,p): return 1-np.sum((t-p)**2)/np.sum((t-t.mean())**2)

# ---- Table II ladder ----
meta=json.load(open("cstar_model.meta.json"))
# mean baseline
mae_mean=np.abs(y-y.mean()).mean()
# self-consistent-c direct
sc=np.array([num(r["c_selfconsistent"]) for r in clean])
mae_sc=np.abs(y-sc).mean(); r2_sc=r2(y,sc)
# ridge LOO
pr=np.zeros(len(y))
for i in range(len(y)):
    m=np.arange(len(y))!=i
    pipe=Pipeline([("s",StandardScaler()),("r",Ridge(alpha=1.0))]); pipe.fit(X[m],y[m]); pr[i]=pipe.predict(X[i:i+1])[0]
mae_ri=np.abs(y-pr).mean(); r2_ri=r2(y,pr)
GAP=3.4  # approx dGap/dc for eV conversion (mean slope)
T2=[("mean c\\* (baseline)", mae_mean, 0.00),
    ("self-consistent c (direct)", mae_sc, r2_sc),
    ("Ridge regression", mae_ri, r2_ri),
    ("Gaussian process", meta["cv_gp"]["loo_mae"], meta["cv_gp"]["loo_r2"]),
    ("**Random forest (primary)**", meta["cv_rf"]["loo_mae"], meta["cv_rf"]["loo_r2"])]

with open("fig_data/tables.md","w",encoding="utf-8") as f:
    # ---- Table II ----
    f.write("### Table II. Leave-one-out model comparison (predicting c\\*, n=%d clean).\n\n"%len(y))
    f.write("| Model | MAE (c\\*) | R² | ~gap MAE (eV) |\n|---|---|---|---|\n")
    for name,mae,rr in T2:
        f.write("| %s | %.3f | %+.2f | %.2f |\n" % (name, mae, rr, mae*GAP))
    f.write("\n")
    # ---- Table III ----
    val={r["formula"]:r for r in csv.DictReader(open("cstar_validation.csv"))}
    diag={r["formula"]:r for r in csv.DictReader(open("cstar_diag.csv"))}
    f.write("### Table III. End-to-end validation on held-out materials (predict c\\* → one mBJ → gap).\n\n")
    f.write("| Material | SG | ML c\\* | mBJ gap (eV) | exp gap (eV) | error (eV) | PBE error (eV) | recovered | true c\\* |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for fo in ("AlN","CuGaS2"):
        v=val[fo]; d=diag[fo]
        rec=100*(1-abs(float(v["gap_err"]))/abs(float(v["pbe_gap_err"])))
        f.write("| %s | %s | %.3f | %.2f | %.2f | %+.2f | %+.2f | %d%% | %.3f |\n" % (
            fo, v["spacegroup"], float(v["c_star_pred"]), float(v["gap_mBJ"]), float(v["ref_gap"]),
            float(v["gap_err"]), float(v["pbe_gap_err"]), round(rec), float(d["c_star"])))
    f.write("\n")
    # ---- Table I (full dataset) ----
    f.write("### Table I. The c\\* dataset (%d materials; %d clean, %d flagged).\n\n"%(len(rows),len(clean),len(rows)-len(clean)))
    f.write("| Formula | Space group | Exp gap (eV) | PBE gap (eV) | c\\* | QC |\n|---|---|---|---|---|---|\n")
    for r in sorted(rows, key=lambda r:(r["qc_pass"]!="1", r["formula"])):
        f.write("| %s | %s | %.2f | %.2f | %.3f | %s |\n" % (
            r["formula"], r["spacegroup"], float(r["ref_gap"]), float(r["mp_pbe_gap"]),
            float(r["c_star"]), r["qc"]))
print("Table II ladder: mean %.3f | selfc %.3f(R2 %.2f) | ridge %.3f(R2 %.2f) | GP %.3f | RF %.3f"
      %(mae_mean,mae_sc,r2_sc,mae_ri,r2_ri,meta["cv_gp"]["loo_mae"],meta["cv_rf"]["loo_mae"]))
print("wrote fig_data/tables.md")
