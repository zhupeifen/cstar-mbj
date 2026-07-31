#!/usr/bin/env python
"""Fixed-c vs material-specific ML-c* benchmark, in GAP space (eV).
Each clean material has a linear E_g(c) from its two probes: gap(c)=gap_lo + m*(c-1.2).
Compare gap error vs experiment for: c=1.2, best single global c, self-consistent c,
and the LOO ML-predicted c*."""
import csv, re, numpy as np
def num(x):
    m=re.search(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?',str(x)); return float(m.group()) if m else np.nan

clean={ (r["formula"],r["spacegroup"]):r for r in csv.DictReader(open("cstar_dataset.csv")) if r["qc_pass"]=="1"}
pred={ (r["formula"],r["spacegroup"]):float(r["pred_cstar"]) for r in csv.DictReader(open("fig_data/materials.csv"))}

gl=[]; gh=[]; ref=[]; sc=[]; mlc=[]
for k,r in clean.items():
    gl.append(float(r["gap_lo"])); gh.append(float(r["gap_hi"])); ref.append(float(r["ref_gap"]))
    sc.append(num(r["c_selfconsistent"])); mlc.append(pred.get(k, float(r["c_star"])))
gl=np.array(gl); gh=np.array(gh); ref=np.array(ref); sc=np.array(sc); mlc=np.array(mlc)
m=(gh-gl)/0.5                                   # slope dGap/dc per material
def gap_at(c): return gl + m*(np.asarray(c)-1.2)   # linear E_g(c)
def mae(c): return np.abs(gap_at(c)-ref).mean()

# best single global c
cs=np.linspace(1.0,2.2,241); errs=[mae(c) for c in cs]; cbest=cs[int(np.argmin(errs))]
n=len(ref)
print("n clean = %d" % n)
print("--- gap MAE vs experiment (eV) ---")
print("  fixed c = 1.20 (common default)   : %.3f" % mae(1.20))
print("  fixed c = 1.30                     : %.3f" % mae(1.30))
print("  best single global c = %.2f        : %.3f" % (cbest, mae(cbest)))
print("  self-consistent c (per material)   : %.3f" % np.abs(gap_at(sc)-ref).mean())
print("  ML material-specific c* (LOO)      : %.3f" % np.abs(gap_at(mlc)-ref).mean())
print("  true c* (fit, sanity ~0)           : %.3f" % np.abs(gap_at(np.array([float(clean[k]['c_star']) for k in clean]))-ref).mean())
# write a small table row set for the manuscript
with open("fig_data/fixedc_benchmark.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["method","gap_MAE_eV"])
    w.writerow(["fixed c = 1.20", round(mae(1.20),3)])
    w.writerow(["best single global c = %.2f"%cbest, round(mae(cbest),3)])
    w.writerow(["self-consistent c (Tran-Blaha)", round(np.abs(gap_at(sc)-ref).mean(),3)])
    w.writerow(["ML material-specific c* (this work, LOO)", round(np.abs(gap_at(mlc)-ref).mean(),3)])
print("wrote fig_data/fixedc_benchmark.csv")
