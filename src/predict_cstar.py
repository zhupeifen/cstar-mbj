#!/usr/bin/env python
"""
predict_cstar.py -- predict the TB-mBJ optimal c* for a material from inexpensive descriptors.

Inputs: the structure (Materials-Project cell, POSCAR), the CHGCAR of one PBEsol SCF on it,
    the PBE fundamental gap, and the vasprun.xml of one PBE DOS run (LORBIT=11) for the band center.
Outputs c* (random forest) and a Gaussian-process sigma. Then run ONE self-consistent TB-mBJ calculation
with CMBJ=c* and check its convergence with check_convergence.py.

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
    # Random forest = primary estimate (as in the paper); GP sigma ranks reliability but is ~1/3 too small.
    print(f"predicted c* = {cstar_rf:.3f}  (random forest)   |  GP {cstar_gp:.3f} +/- {sig:.3f}  (sigma x1.4 ~ {1.4*sig:.3f})")
    print(f"  -> run one self-consistent TB-mBJ calculation with CMBJ = {cstar_rf:.3f}, then check_convergence.py\n")

if __name__=="__main__": main()
