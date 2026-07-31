#!/usr/bin/env python
"""
predict_cstar.py -- predict the TB-mBJ optimal c* for a material from a CHEAP PBE run.

Inputs (all from one PBE relaxation + SCF that writes CHGCAR):
    a relaxed structure (CONTCAR/POSCAR), the PBE CHGCAR, and the PBE fundamental gap.
Outputs c* (+ GP uncertainty). Then run ONE TB-mBJ SCF with CMBJ=c* to get a near-experimental gap.

Usage:
    predict_cstar.py --contcar CONTCAR --chgcar CHGCAR --pbe-gap 0.61
    predict_cstar.py --rundir runs_ml/Si_Fd-3m/0_relax --vasprun vasprun.xml   # auto gap
"""
import sys, os, argparse, joblib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_cstar_dataset import descriptors, grad_ratio_from_chgcar, anion_p_center
from pymatgen.core import Structure

def pbe_gap_from_vasprun(path):
    from pymatgen.io.vasp.outputs import Vasprun
    v=Vasprun(path, parse_dos=False, parse_eigen=True)
    return float(v.eigenvalue_band_properties[0])   # (gap, cbm, vbm, is_direct)[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rundir", default=".", help="folder holding CONTCAR/CHGCAR (default .)")
    ap.add_argument("--contcar"); ap.add_argument("--chgcar"); ap.add_argument("--vasprun")
    ap.add_argument("--dos-vasprun", help="vasprun.xml of a PBE DOS run (LORBIT=11) for the band_center feature")
    ap.add_argument("--pbe-gap", type=float, help="PBE fundamental gap (eV); or use --vasprun")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","model","cstar_model.joblib"))
    a=ap.parse_args()
    contcar=a.contcar or os.path.join(a.rundir,"CONTCAR")
    chgcar =a.chgcar  or os.path.join(a.rundir,"CHGCAR")
    gap = a.pbe_gap
    if gap is None:
        vr=a.vasprun or os.path.join(a.rundir,"vasprun.xml")
        if not os.path.exists(vr): sys.exit("need --pbe-gap or a vasprun.xml (--vasprun/rundir)")
        gap=pbe_gap_from_vasprun(vr)
    struct=Structure.from_file(contcar)
    grad=grad_ratio_from_chgcar(chgcar)
    d=descriptors(struct, gap, grad); d["mp_pbe_gap"]=gap        # pbe_gap == mp_pbe_gap (the PBE gap)
    B=joblib.load(a.model); feats=B["meta"]["features"]; gp=B["model"]; rf=B["rf"]
    if "band_center" in feats:                                   # needs a PBE DOS run (LORBIT=11)
        dv=a.dos_vasprun or os.path.join(a.rundir,"dos","vasprun.xml")
        if not os.path.exists(dv):
            sys.exit("model uses band_center: pass --dos-vasprun (a PBE DOS run, LORBIT=11) or put it at <rundir>/dos/vasprun.xml")
        d["band_center"]=anion_p_center(dv)
    x=np.array([[float(d[f]) for f in feats]])
    cstar,sigma=gp.predict(x, return_std=True); cstar_gp=float(cstar[0]); sig=float(sigma[0])
    cstar_rf=float(rf.predict(x)[0])
    print(f"\nmaterial: {struct.composition.reduced_formula}   PBE gap = {gap:.2f} eV   grad_ratio = {grad:.3f}")
    # GP is the primary point estimate (best on the 128-material set: LOO MAE 0.100, R^2 0.55)
    # and supplies the uncertainty sigma; RF is a cross-check.
    print(f"predicted c* = {cstar_gp:.3f} +/- {sig:.3f}  (GP, primary)   |  {cstar_rf:.3f}  (RF, cross-check)")
    print(f"  -> run one TB-mBJ SCF with CMBJ = {cstar_gp:.3f}  (build_pipeline --cstar / tune_cmbj probe)\n")

if __name__=="__main__": main()
