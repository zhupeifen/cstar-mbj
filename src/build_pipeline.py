#!/usr/bin/env python
"""
build_pipeline.py  --  META-GGA (TB-mBJ) band-structure / DOS / optics pipeline generator.

Given a POSCAR (e.g. fetched from Materials Project), generate a full staged VASP
workflow that reproduces the Cs2AgInCl6 / CsGeBr3 mBJ recipe on the orca cluster:

  0_relax     PBE cell+ion relaxation (ISIF=3)                    -> CONTCAR
  1_mbj_scf   master TB-mBJ SCF (self-consistent c)               -> CHGCAR + WAVECAR + IBZKPT
                                                                     + ELFCAR/AECCAR/LOCPOT
  2_bands     self-consistent mBJ on HYBRID mesh (IBZKPT weighted
              + zero-weight high-symmetry path), FIXED CMBJ=<c>    -> EIGENVAL + PROCAR (LORBIT=12)
  3_dos       mBJ DOS, restart master WAVECAR+CHGCAR, FIXED CMBJ   -> DOSCAR (LORBIT=12)
  4_optics    mBJ optics (LOPTICS), read master, FIXED CMBJ       -> OUTCAR dielectric

KEY LESSONS baked in (from the CsGeBr3 campaign, see memory csgebr3-mbj-recalc):
  * On this cluster mBJ needs LMIXTAU=F (omit), ISPIN=1 (non-magnetic), LMAXMIX=4, ALGO=Damped.
  * The self-consistent CMBJ c-parameter integral intermittently blows to NaN in the
    property runs. FIX: let the *master* compute c self-consistently, then PIN that value
    (CMBJ=<c>) in bands/dos/optics so the unstable integral is never re-evaluated.  <-- filled after master
  * Bands need the converged mBJ density/tau: use the hybrid mesh (IBZKPT weighted points
    reproduce the master density -> gap opens) with ICHARG=1; ICHARG=11 / fresh-WF => gapless.
  * DOS restarts from the master WAVECAR+CHGCAR on the SAME mesh (ISTART=1) => converges instantly.

The bands KPOINTS is assembled ON THE CLUSTER after the master run (needs its IBZKPT);
this script emits the zero-weight path portion + a small assembler.
"""
import os, sys, json, argparse, subprocess, re
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

# ---- per-element PAW potential (PBE_54, MP-style) and RWIGS (Angstrom, ~atomic radius) ----
POTCAR_MAP = {
 'H':'H','Li':'Li_sv','Be':'Be','B':'B','C':'C','N':'N','O':'O','F':'F',
 'Na':'Na_pv','Mg':'Mg','Al':'Al','Si':'Si','P':'P','S':'S','Cl':'Cl',
 'K':'K_sv','Ca':'Ca_sv','Sc':'Sc_sv','Ti':'Ti_pv','V':'V_pv','Cr':'Cr_pv',
 'Mn':'Mn_pv','Fe':'Fe','Co':'Co','Ni':'Ni','Cu':'Cu','Zn':'Zn','Ga':'Ga_d',
 'Ge':'Ge_d','As':'As','Se':'Se','Br':'Br',
 'Rb':'Rb_sv','Sr':'Sr_sv','Y':'Y_sv','Zr':'Zr_sv','Nb':'Nb_pv','Mo':'Mo_pv',
 'Ag':'Ag','Cd':'Cd','In':'In_d','Sn':'Sn_d','Sb':'Sb','Te':'Te','I':'I',
 'Cs':'Cs_sv','Ba':'Ba_sv','La':'La','Ce':'Ce','Eu':'Eu','Gd':'Gd','Tb':'Tb',
 'Hf':'Hf_pv','Ta':'Ta_pv','W':'W_pv','Pt':'Pt','Au':'Au','Hg':'Hg',
 'Tl':'Tl_d','Pb':'Pb_d','Bi':'Bi_d',
}
RWIGS = {
 'H':0.37,'Li':1.55,'B':0.85,'C':0.77,'N':0.75,'O':0.82,'F':0.71,
 'Na':1.60,'Mg':1.45,'Al':1.25,'Si':1.10,'P':1.05,'S':1.05,'Cl':1.02,
 'K':2.00,'Ca':1.80,'Ti':1.47,'Mn':1.35,'Fe':1.30,'Cu':1.28,'Zn':1.35,'Ga':1.40,
 'Ge':1.22,'As':1.20,'Se':1.20,'Br':1.14,
 'Rb':2.15,'Sr':1.95,'Y':1.80,'Zr':1.60,'Ag':1.45,'Cd':1.50,'In':1.55,'Sn':1.45,
 'Sb':1.45,'Te':1.40,'I':1.40,
 'Cs':2.35,'Ba':2.15,'La':1.95,'Eu':1.85,'Tb':1.75,
 'Tl':1.55,'Pb':1.55,'Bi':1.55,
}
# valence electron count per PAW symbol (PBE_54) -> NELECT -> NBANDS
ZVAL = {
 'H':1,'Li_sv':3,'Be':2,'B':3,'C':4,'N':5,'O':6,'F':7,
 'Na_pv':7,'Mg':2,'Al':3,'Si':4,'P':5,'S':6,'Cl':7,
 'K_sv':9,'Ca_sv':10,'Sc_sv':11,'Ti_pv':10,'V_pv':11,'Cr_pv':12,
 'Mn_pv':13,'Fe':8,'Co':9,'Ni':10,'Cu':11,'Zn':12,'Ga_d':13,
 'Ge_d':14,'As':5,'Se':6,'Br':7,
 'Rb_sv':9,'Sr_sv':10,'Y_sv':11,'Zr_sv':12,'Nb_pv':11,'Mo_pv':12,
 'Ag':11,'Cd':12,'In_d':13,'Sn_d':14,'Sb':5,'Te':6,'I':7,
 'Cs_sv':9,'Ba_sv':10,'La':11,'Ce':12,'Eu':17,'Gd':18,'Tb':19,
 'Hf_pv':10,'Ta_pv':11,'W_pv':12,'Pt':10,'Au':11,'Hg':12,
 'Tl_d':13,'Pb_d':14,'Bi_d':15,
}
def rwigs(el): return RWIGS.get(el, 1.35)
def potsym(el): return POTCAR_MAP.get(el, el)
def zval(sym):  return ZVAL.get(sym, 8)

def kmesh(struct, rk=40):
    """Gamma-centered divisions from a k-point length parameter Rk (~40 dense)."""
    recip = struct.lattice.reciprocal_lattice.abc  # 2pi/a already included
    return [max(1, int(round(rk * b / (2*np.pi)))) for b in recip]

def kmesh_line(divs):
    return "Automatic mesh\n0\nGamma\n  %d %d %d\n  0 0 0\n" % tuple(divs)

# ---------------- INCAR templates ----------------
def incar_relax(name, elems, encut, gga=None):
    # gga=None -> PBE ; gga='PS' -> PBEsol (better solid-state lattice constants)
    tag = {"PS": "PBEsol", None: "PBE"}.get(gga, gga)
    ggaline = f"   GGA = {gga}\n" if gga else ""
    return f"""SYSTEM = {name} {tag} relax (ISIF=3)
ISTART = 0
{ggaline}  PREC = Accurate
 ENCUT = {encut}
 LREAL = .FALSE.
 NCORE = 9
ISMEAR = 0
 SIGMA = 0.05
 ISPIN = 1
LMAXMIX = 4
IBRION = 2
  ISIF = 3
   NSW = 100
 EDIFF = 1.0e-06
EDIFFG = -0.01
 LWAVE = .FALSE.
LCHARG = .FALSE.
"""

def incar_master(name, elems, encut, cmbj=None):
    rw = " ".join("%.2f" % rwigs(e) for e in elems)
    # cmbj is None -> self-consistent c (original behavior, extracted downstream after this run).
    # cmbj given   -> FIXED seed (e.g. ML-predicted c*); skips the NaN-prone self-consistent integral.
    if cmbj is None:
        sysline  = f"SYSTEM = {name} master TB-mBJ SCF (self-consistent c -> writes CHGCAR/WAVECAR/IBZKPT/ELFCAR)"
        cmbj_ln  = ""
    else:
        sysline  = f"SYSTEM = {name} master TB-mBJ SCF (SEEDED FIXED CMBJ={cmbj:.4f} -> writes CHGCAR/WAVECAR/IBZKPT/ELFCAR)"
        cmbj_ln  = f"  CMBJ = {cmbj:.4f}\n"
    return f"""{sysline}
ISTART = 0
  PREC = Accurate
 ENCUT = {encut}
 LREAL = .FALSE.
 NCORE = 9
METAGGA = MBJ
{cmbj_ln} LASPH = .TRUE.
LMAXMIX = 4
  ALGO = Damped
  TIME = 0.40
  NELM = 250
ICHARG = 2
ISMEAR = 0
 SIGMA = 0.05
 ISPIN = 1
IBRION = -1
   NSW = 0
 EDIFF = 1.0e-05
LORBIT = 11
 RWIGS = {rw}
 LWAVE = .TRUE.
LCHARG = .TRUE.
 LELF = .TRUE.
LAECHG = .TRUE.
 LVTOT = .TRUE.
LVHAR = .TRUE.
"""

def incar_bands(name, elems, encut):
    rw = " ".join("%.2f" % rwigs(e) for e in elems)
    return f"""SYSTEM = {name} mBJ bands LORBIT=12 (hybrid mesh, ICHARG=1 read master, FIXED CMBJ)
ISTART = 0
  PREC = Accurate
 ENCUT = {encut}
 LREAL = .FALSE.
 NCORE = 9
METAGGA = MBJ
  CMBJ = __CMBJ__
 LASPH = .TRUE.
LMAXMIX = 4
  ALGO = Damped
  TIME = 0.40
  NELM = 250
ICHARG = 1
ISMEAR = 0
 SIGMA = 0.05
 ISPIN = 1
IBRION = -1
   NSW = 0
 EDIFF = 1.0e-05
LORBIT = 12
 RWIGS = {rw}
NBANDS = __NBANDS__
 LWAVE = .FALSE.
LCHARG = .FALSE.
"""

def incar_dos(name, elems, encut):
    rw = " ".join("%.2f" % rwigs(e) for e in elems)
    return f"""SYSTEM = {name} mBJ DOS LORBIT=12 (restart master WAVECAR+CHGCAR, FIXED CMBJ)
ISTART = 1
  PREC = Accurate
 ENCUT = {encut}
 LREAL = .FALSE.
 NCORE = 9
METAGGA = MBJ
  CMBJ = __CMBJ__
 LASPH = .TRUE.
LMAXMIX = 4
  ALGO = Damped
  TIME = 0.40
  NELM = 250
ICHARG = 1
ISMEAR = 0
 SIGMA = 0.10
 ISPIN = 1
IBRION = -1
   NSW = 0
 EDIFF = 1.0e-05
LORBIT = 12
 RWIGS = {rw}
NBANDS = __NBANDS__
 NEDOS = 2500
 LWAVE = .FALSE.
LCHARG = .FALSE.
"""

def incar_optics(name, elems, encut):
    return f"""SYSTEM = {name} mBJ optics (LOPTICS, read master, FIXED CMBJ)
ISTART = 1
  PREC = Accurate
 ENCUT = {encut}
 LREAL = .FALSE.
 NCORE = 9
METAGGA = MBJ
  CMBJ = __CMBJ__
 LASPH = .TRUE.
LMAXMIX = 4
  ALGO = Damped
  TIME = 0.40
  NELM = 250
ICHARG = 1
ISMEAR = 0
 SIGMA = 0.05
 ISPIN = 1
IBRION = -1
   NSW = 0
 EDIFF = 1.0e-05
LOPTICS = .TRUE.
CSHIFT = 0.10
NBANDS = __NBANDS__
 NEDOS = 3000
 LWAVE = .FALSE.
LCHARG = .FALSE.
"""

SLURM = """#!/bin/bash -l
#SBATCH -J {jobname}
#SBATCH -A missouri
#SBATCH -p bw
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -t {walltime}
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
sd=$SLURM_SUBMIT_DIR; wd=/scratch/$USER/$SLURM_JOB_ID; mkdir -p $wd
cp $sd/{{INCAR,KPOINTS,POTCAR,POSCAR}} $wd/ 2>/dev/null
{restart}
cd $wd
module load mpich/3.4.2-ofi; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
time prun vasp
{collect}

# Remove the compute workdir once the run is verifiably complete and nothing was
# truncated on the way back. Without this every stage leaves a workdir behind:
# 341 orphans had reached 433 GB on /scratch/pzhu by 2026-07-28, nearly all of it
# WAVECAR (four Cs2AgInCl6 jobs at ~78 GB each). Files not in the collect list
# are skipped by the -e test, so they do not block cleanup.
cleanup_ok=1
grep -q "General timing" OUTCAR 2>/dev/null || cleanup_ok=0
if [[ $cleanup_ok -eq 1 ]]; then
  for f in *; do
    [[ -f "$f" && -e "$sd/$f" ]] || continue
    if [[ $(stat -c%s "$f") -ne $(stat -c%s "$sd/$f") ]]; then
      echo "truncated copy-back: $f"; cleanup_ok=0
    fi
  done
fi
if [[ $cleanup_ok -eq 1 ]]; then
  cd /; rm -rf "$wd"; echo "workdir $wd removed"
else
  echo "WARNING: run incomplete or copy-back suspect - keeping $wd"
fi

touch $sd/STAGE_DONE; exit 0
"""

def slurm(jobname, walltime, restart="", collect=None):
    if collect is None:
        collect = ('for f in OUTCAR OSZICAR CONTCAR vasprun.xml DOSCAR EIGENVAL PROCAR '
                   'IBZKPT ELFCAR AECCAR0 AECCAR2 LOCPOT CHGCAR WAVECAR; do '
                   '[[ -s $f ]] && cp $f $sd/; done')
    return SLURM.format(jobname=jobname, walltime=walltime, restart=restart, collect=collect)

def high_sym_path(struct, line_density=20, pts_per_seg=None):
    """Zero-weight band path + labels, with consecutive-duplicate k-points removed.
    pymatgen duplicates shared segment endpoints; dedup gives a clean continuous polyline
    (true discontinuities like X|M remain as adjacent distinct points -> MATLAB merges them).
    pts_per_seg (if set) forces a FIXED number of k-points per high-symmetry segment
    (endpoints labeled, interior blank), overriding line_density."""
    kp = HighSymmKpath(struct, symprec=0.01)
    if pts_per_seg:
        kk = kp.kpath['kpoints']; branches = kp.kpath['path']
        kpts = []; labels = []
        for br in branches:
            for i in range(len(br) - 1):
                a = np.array(kk[br[i]], float); b = np.array(kk[br[i+1]], float)
                for j in range(pts_per_seg):
                    t = j / (pts_per_seg - 1.0)
                    kpts.append(a + (b - a) * t)
                    labels.append(br[i] if j == 0 else (br[i+1] if j == pts_per_seg - 1 else None))
    else:
        kpts, labels = kp.get_kpoints(line_density=line_density, coords_are_cartesian=False)
    dk, dl = [], []
    for k, lab in zip(kpts, labels):
        if dk and np.allclose(k, dk[-1], atol=1e-6):
            if lab and not dl[-1]:           # carry the label onto the kept duplicate
                dl[-1] = lab
            continue
        dk.append(np.array(k, float)); dl.append(lab)
    return dk, dl, kp.kpath

def write_bandstructure_data(labels, outpath):
    """MATLAB banddos pipeline label file: symmetryPoints {L0 0 L1 n1 L2 n2 ...}."""
    # indices of labelled points
    idx = [(i, lab) for i, lab in enumerate(labels) if lab]
    # merge coincident (same index adjacent) -> keep unique breakpoints
    merged = []
    for i, lab in idx:
        if merged and i - merged[-1][0] == 0:
            continue
        merged.append((i, lab))
    # cumulative increments; first label increment 0.
    # readbandstructure needs SINGLE-CHAR labels (S is a char row split by cellstr per char) and
    # treats digits in a label as k-point increments. So subscripted points (B_1, P_1) must map to
    # a single UNUSED uppercase letter; the true LaTeX name is emitted to a companion label-map file
    # for the plotter to restore on display.
    import re as _re
    used = set()
    dispmap = {}   # placeholder-letter -> LaTeX display string
    def base_label(lab):
        L = lab.replace("\\Gamma", "G").replace("$\\Gamma$", "G").replace("GAMMA", "G")
        L = L.replace("\\", "").replace("$", "")
        if L in ("Gamma", "\\Gamma"): L = "G"
        return L
    # pre-scan: reserve the plain (digit-free) single letters first
    for _, lab in merged:
        L = base_label(lab)
        if not _re.search(r"\d", L) and len(L) == 1:
            used.add(L)
    def alias(L):
        m = _re.match(r"^([A-Za-z]+)_?(\d+)$", L)
        if not m:
            return L  # already a clean single/simple letter
        stem, sub = m.group(1), m.group(2)
        for c in "YWJKQZXVUTRNMOHIEDCA":     # candidate unused letters
            if c not in used:
                used.add(c); dispmap[c] = "$%s_{%s}$" % (stem, sub); return c
        return stem[0]
    toks = []
    for j, (i, lab) in enumerate(merged):
        L = base_label(lab)
        if _re.search(r"\d", L):
            L = alias(L)
        toks.append("%s 0" % L if j == 0 else "%s %d" % (L, i - merged[j-1][0]))
    n = len(merged)
    with open(outpath, "w") as f:
        f.write("nSymmetryPoints\t%d\n" % n)
        f.write("symmetryPoints\t{%s}\n" % " ".join(toks))
    if dispmap:
        with open(os.path.join(os.path.dirname(outpath) or ".", "label_map.txt"), "w") as f:
            for c, tex in dispmap.items():
                f.write("%s\t%s\n" % (c, tex))
    return merged

def path_kpoints_block(kpts, labels):
    """VASP 'Reciprocal' zero-weight explicit block (list of lines, no header)."""
    lines = []
    for k in kpts:
        lines.append(" %12.8f %12.8f %12.8f   0" % (k[0], k[1], k[2]))
    return lines

def predict_cstar_seed(seed_relax, pbe_gap=None, model=None):
    """Predict TB-mBJ c* from a completed PBE relax dir via ml/predict_cstar.py.
    seed_relax must hold CONTCAR + CHGCAR (+ vasprun.xml, unless pbe_gap given).
    Returns the GP-predicted c* (float). Reuses the deployed model = single source of truth."""
    here = os.path.dirname(os.path.abspath(__file__))
    pred = os.path.join(here, "ml", "predict_cstar.py")
    if not os.path.exists(pred):
        sys.exit("cannot seed c*: predict_cstar.py not found at %s" % pred)
    cmd = [sys.executable, pred, "--rundir", seed_relax]
    if pbe_gap is not None:
        cmd += ["--pbe-gap", "%.6f" % pbe_gap]
    if model:
        cmd += ["--model", model]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("predict_cstar.py failed while seeding c*:\n%s%s" % (out.stdout, out.stderr))
    m = re.search(r"predicted c\*\s*=\s*([0-9.]+)", out.stdout)
    if not m:
        sys.exit("could not parse c* from predict_cstar.py output:\n%s" % out.stdout)
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poscar", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--encut", type=int, default=500)
    ap.add_argument("--encut-relax", type=int, default=600)
    ap.add_argument("--rk", type=int, default=45, help="k-point length param (dense mesh)")
    ap.add_argument("--rk-scf", type=int, default=36, help="k-point length param (master/dos mesh)")
    ap.add_argument("--line-density", type=int, default=20)
    ap.add_argument("--pts-per-seg", type=int, default=None, help="fixed k-points per band segment (overrides line-density)")
    ap.add_argument("--potcar-dir", default=None, help="root of PBE PAW library (contains <sym>/POTCAR)")
    ap.add_argument("--primitive", action="store_true", help="reduce to primitive cell first")
    ap.add_argument("--relax-gga", default="PS", help="relax functional: 'PS'=PBEsol (default), 'PE'/none=PBE")
    ap.add_argument("--cstar", type=float, default=None,
                    help="seed a FIXED mBJ c* into master+downstream (skips the NaN-prone self-consistent c)")
    ap.add_argument("--seed-relax", default=None,
                    help="dir of a completed PBE relax (CONTCAR+CHGCAR[+vasprun.xml]); predict c* via ml/predict_cstar.py")
    ap.add_argument("--seed-gap", type=float, default=None,
                    help="PBE fundamental gap (eV) for --seed-relax prediction (else read its vasprun.xml)")
    ap.add_argument("--cstar-model", default=None, help="override model .joblib for --seed-relax prediction")
    a = ap.parse_args()

    # resolve an optional c* seed: explicit --cstar wins, else predict from --seed-relax
    cstar = a.cstar
    if cstar is None and a.seed_relax:
        cstar = predict_cstar_seed(a.seed_relax, a.seed_gap, a.cstar_model)
        print("SEED c* (ML-predicted) = %.4f   from %s" % (cstar, a.seed_relax))
    elif cstar is not None:
        print("SEED c* (explicit)     = %.4f" % cstar)

    st = Structure.from_file(a.poscar)
    sga = SpacegroupAnalyzer(st, symprec=0.01)
    st = sga.get_primitive_standard_structure() if a.primitive else sga.get_conventional_standard_structure()
    sg = sga.get_space_group_symbol()

    elems = [str(s) for s in st.composition.elements]        # unique, comp order
    # NELECT from PAW ZVAL -> generous NBANDS (multiple of NCORE=9) for conduction/optics
    nelect = int(round(sum(zval(potsym(str(sp))) * st.composition[sp] for sp in st.composition)))
    nbands = int(np.ceil(max(2.0*nelect, nelect + 4*len(st)) / 9.0) * 9)
    os.makedirs(a.outdir, exist_ok=True)
    st.to(fmt="poscar", filename=os.path.join(a.outdir, "POSCAR"))
    os.makedirs(os.path.join(a.outdir, "0_relax"), exist_ok=True)
    st.to(fmt="poscar", filename=os.path.join(a.outdir, "0_relax", "POSCAR"))  # relax starting cell

    divs_scf  = kmesh(st, a.rk_scf)
    divs_dos  = kmesh(st, a.rk_scf)
    divs_opt  = kmesh(st, a.rk_scf)

    # ---- k-path (zero weight) ----
    kpts, labels, kpath = high_sym_path(st, a.line_density, a.pts_per_seg)
    matlab_dir = os.path.join(a.outdir, "matlab"); os.makedirs(matlab_dir, exist_ok=True)
    merged = write_bandstructure_data(labels, os.path.join(matlab_dir, "Bandstructure.DATA"))
    path_lines = path_kpoints_block(kpts, labels)
    with open(os.path.join(a.outdir, "2_bands", "KPATH_zeroweight.txt"), "w") if False else open(os.devnull,"w"): pass

    R_MASTER_POS = "cp $sd/../0_relax/CONTCAR $wd/POSCAR 2>/dev/null   # use relaxed cell"
    R_CHG        = "cp $sd/../1_mbj_scf/CHGCAR $wd/ 2>/dev/null"
    R_CHG_WAV    = "cp $sd/../1_mbj_scf/CHGCAR $wd/ 2>/dev/null; cp $sd/../1_mbj_scf/WAVECAR $wd/ 2>/dev/null"
    stages = {
        "0_relax":   (incar_relax(a.name, elems, a.encut_relax, a.relax_gga), kmesh_line(divs_scf),
                      slurm(a.name[:8]+"_rlx", "04:00:00")),
        "1_mbj_scf": (incar_master(a.name, elems, a.encut, cmbj=cstar), kmesh_line(divs_scf),
                      slurm(a.name[:8]+"_scf", "06:00:00", restart=R_MASTER_POS)),
        "2_bands":   (incar_bands(a.name, elems, a.encut),       None,   # KPOINTS built on cluster
                      slurm(a.name[:8]+"_bnd", "04:00:00", restart=R_CHG)),
        "3_dos":     (incar_dos(a.name, elems, a.encut),         kmesh_line(divs_dos),
                      slurm(a.name[:8]+"_dos", "03:00:00", restart=R_CHG_WAV)),
        "4_optics":  (incar_optics(a.name, elems, a.encut),      kmesh_line(divs_opt),
                      slurm(a.name[:8]+"_opt", "04:00:00", restart=R_CHG_WAV)),
    }
    for stg, (inc, kp, sl) in stages.items():
        inc = inc.replace("__NBANDS__", str(nbands))
        if cstar is not None:
            inc = inc.replace("__CMBJ__", "%.4f" % cstar)     # seed all stages now; skip post-master extract
        # else: __CMBJ__ left as placeholder, filled on cluster from master OUTCAR (self-consistent path)
        d = os.path.join(a.outdir, stg); os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "INCAR"), "w").write(inc)
        if kp: open(os.path.join(d, "KPOINTS"), "w").write(kp)
        open(os.path.join(d, "job.slurm"), "w").write(sl)

    # zero-weight path -> file the cluster assembler prepends IBZKPT to
    with open(os.path.join(a.outdir, "2_bands", "KPATH_zeroweight.txt"), "w") as f:
        f.write("\n".join(path_lines) + "\n")
    open(os.path.join(a.outdir, "2_bands", "npath.txt"), "w").write(str(len(path_lines)))

    # POTCAR: symbol list + (optionally) assemble locally
    syms = [potsym(e) for e in elems]
    open(os.path.join(a.outdir, "POTCAR_symbols.txt"), "w").write("\n".join(syms) + "\n")
    potcar_written = False
    if a.potcar_dir:
        chunks = []
        for s in syms:
            p = os.path.join(a.potcar_dir, s, "POTCAR")
            if not os.path.exists(p):
                print("WARN missing POTCAR for %s at %s" % (s, p)); chunks = None; break
            chunks.append(open(p).read())
        if chunks:
            pot = "".join(chunks)
            for stg in stages:
                open(os.path.join(a.outdir, stg, "POTCAR"), "w").write(pot)
            potcar_written = True

    meta = dict(name=a.name, spacegroup=sg, elements=elems, potcar_symbols=syms,
                rwigs=[rwigs(e) for e in elems], encut=a.encut,
                kmesh_scf=divs_scf, npath=len(path_lines), nelect=nelect, nbands=nbands,
                nsites=len(st), formula=st.composition.reduced_formula,
                path_labels=[m[1] for m in merged],
                cmbj_seed=(round(cstar, 4) if cstar is not None else None),
                cmbj_seed_source=("explicit" if a.cstar is not None else
                                  ("ml:%s" % a.seed_relax) if a.seed_relax else "self-consistent"))
    json.dump(meta, open(os.path.join(a.outdir, "pipeline_meta.json"), "w"), indent=2)
    # marker: presence => all CMBJ already fixed, cluster should NOT run extract_cmbj.sh after master
    if cstar is not None:
        open(os.path.join(a.outdir, "cmbj_seed.txt"), "w").write("%.4f\n" % cstar)

    # normalize ALL generated files to LF (SLURM + VASP reject CRLF on Windows-written files)
    for root, _, files in os.walk(a.outdir):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                b = open(p, "rb").read()
                if b"\r\n" in b: open(p, "wb").write(b.replace(b"\r\n", b"\n"))
            except Exception: pass

    print("=== built %s (%s, SG=%s, %d atoms) ===" % (a.name, meta["formula"], sg, len(st)))
    print("elements:", elems, " POTCAR:", syms, " (written=%s)" % potcar_written)
    print("NELECT:", nelect, " NBANDS:", nbands)
    print("SCF mesh:", divs_scf, " zero-weight path pts:", len(path_lines))
    print("Bandstructure.DATA:", open(os.path.join(matlab_dir,"Bandstructure.DATA")).read().strip())

if __name__ == "__main__":
    main()
