#!/usr/bin/env python
"""
gen_cstar_dataset.py -- build an ML training set for predicting the OPTIMAL TB-mBJ c-parameter (c*).

For each material the driver runs the (cheap) pipeline and emits ONE row:

    {cheap descriptors}  ->  c*   (the mBJ c that reproduces the reference gap)

so a model f(descriptors) -> c* can later give near-GW/experimental gaps at mBJ cost, with a
single mBJ run and no reference. Labels are produced automatically: mBJ is ~linear in c
(dGap/dc ~ 1.8-2.0 eV), so two gap-probes bracket the reference gap and `tune_cmbj.fit` solves c*.

Per material
------------
  1. fetch structure (mp_fetch)                                  [cheap]
  2. PBEsol relax                                                [1 DFT job]
  3. PBE SCF -> PBE gap + CHGCAR (for <|grad rho|/rho>)          [1 DFT job]
  4. cheap DESCRIPTORS from the relaxed structure + PBE density  [local, no DFT]
  5. two mBJ gap-probes at c_lo, c_hi (hybrid mesh, fixed c)     [2 DFT jobs]
  6. c* = tune_cmbj.fit([(c_lo,g_lo),(c_hi,g_hi)], ref_gap)      [local]
  7. append the row to the dataset CSV

Input CSV (materials.csv):  formula,spacegroup,ref_gap[,mpid]
   e.g.   CsPbBr3,Pm-3m,2.30
          CsGeBr3,R3m,2.32

The DFT submit/wait is abstracted in `run_stage()` (SLURM on orca, mirrors the pipeline
orchestration). Descriptors + c* math are fully implemented below.
"""
import os, csv, json, argparse, subprocess, sys
import numpy as np
from pymatgen.core import Structure, Element, Composition

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_cmbj, tune_cmbj                              # reuse the pipeline tools

# ----------------------------- cheap descriptors (no mBJ) -----------------------------
def descriptors(struct: Structure, pbe_gap: float, grad_ratio: float) -> dict:
    """All features are cheap: composition/structure + one PBE density gradient + PBE gap."""
    comp = struct.composition
    els = list(comp.elements)
    frac = [comp.get_atomic_fraction(e) for e in els]
    X    = [e.X or 0.0 for e in els]                        # Pauling electronegativity
    Z    = [e.Z for e in els]
    row  = struct.composition.average_electroneg
    d = dict(
        n_species        = len(els),
        natoms           = len(struct),
        volume_per_atom  = struct.volume / len(struct),
        density          = struct.density,
        mean_electroneg  = float(np.average(X, weights=frac)),
        electroneg_spread= float(max(X) - min(X)),           # ionicity proxy
        mean_Z           = float(np.average(Z, weights=frac)),
        max_Z            = float(max(Z)),                    # SOC proxy (heavy elements)
        mean_row         = float(np.average([e.row for e in els], weights=frac)),
        valence_e_per_atom = comp.total_electrons / len(struct),
        grad_ratio       = grad_ratio,                       # <|grad rho|/rho> from PBE CHGCAR (1/A)
        c_selfconsistent = -0.012 + 1.023 * np.sqrt(grad_ratio * 0.52917721067),  # TB-mBJ formula
        pbe_gap          = pbe_gap,
    )
    return d

# ----------------------------- HPC (orca) submit / wait / fetch -----------------------------
import time
HOST        = os.environ.get("MBJ_HOST", "pzhu@medea.orca.offn.onenet.net")
REMOTE_BASE = os.environ.get("MBJ_REMOTE", "/scratch/pzhu/ml_cstar")
SSH_OPTS    = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
_NOISE = ("WARNING", "vulnerable", "post-quantum", "openssh", "upgraded", "store now")

def _clean(s):
    return "\n".join(l for l in s.splitlines() if not any(n.lower() in l.lower() for n in _NOISE))

def _ssh(cmd):
    p = subprocess.run(["ssh"] + SSH_OPTS + [HOST, cmd], capture_output=True, text=True)
    return _clean(p.stdout).strip()

def _scp(src, dst):
    subprocess.run(["scp"] + SSH_OPTS + ["-r", src, dst], capture_output=True, text=True)

def _lf(local_dir):
    for root, _, files in os.walk(local_dir):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                b = open(p, "rb").read()
                if b"\r\n" in b: open(p, "wb").write(b.replace(b"\r\n", b"\n"))
            except Exception: pass

JOBSLURM_ORCA = """#!/bin/bash -l
#SBATCH -J {job}
#SBATCH -A missouri
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -t {wall}
#SBATCH -o slurm-%j.out
sd=$SLURM_SUBMIT_DIR; wd=/scratch/$USER/$SLURM_JOB_ID; mkdir -p $wd
cp $sd/{{INCAR,KPOINTS,POTCAR,POSCAR}} $wd/ 2>/dev/null
{restart}
cd $wd
module load mpich/3.4.2-ofi; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
time prun vasp
for f in OUTCAR OSZICAR CONTCAR EIGENVAL DOSCAR CHGCAR IBZKPT vasprun.xml; do [[ -s $f ]] && cp $f $sd/; done
touch $sd/STAGE_DONE; exit 0
"""

# Hellbender (MU) port: general/requeue, node-local scratch, VASP 6.4.2, MKL-thread pin (critical).
JOBSLURM_HELLBENDER = """#!/bin/bash -l
#SBATCH -J {job}
#SBATCH -A general
#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -t {wall}
#SBATCH -o slurm-%j.out
sd=$SLURM_SUBMIT_DIR; wd=/local/scratch/$USER/$SLURM_JOB_ID; mkdir -p $wd
cp $sd/{{INCAR,KPOINTS,POTCAR,POSCAR}} $wd/ 2>/dev/null
{restart}
cd $wd
module purge; module load vasp/6.4.2_gcc_12.3.0_openmpi_4.1.5
export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1
time srun vasp_std
for f in OUTCAR OSZICAR CONTCAR EIGENVAL DOSCAR CHGCAR IBZKPT vasprun.xml; do [[ -s $f ]] && cp $f $sd/; done
touch $sd/STAGE_DONE; exit 0
"""

JOBSLURM = JOBSLURM_HELLBENDER if os.environ.get("MBJ_CLUSTER","orca").lower()=="hellbender" else JOBSLURM_ORCA

def run_stage(local_dir, remote_sub, fetch=(), wall="02:00:00", jobname="mbj",
              restart="", poll=30, max_polls=600):
    """Transfer local_dir -> REMOTE_BASE/remote_sub, submit job.slurm (auto-written if absent),
    poll for STAGE_DONE, scp `fetch` files back into local_dir. Returns job id or None on timeout."""
    remote = "%s/%s" % (REMOTE_BASE, remote_sub)
    # ALWAYS write our cluster-correct job.slurm -- build_pipeline drops its own (orca -p bw) one
    # into each stage dir, and a stale/wrong-partition job.slurm would silently break submission.
    open(os.path.join(local_dir, "job.slurm"), "w").write(
        JOBSLURM.format(job=jobname, wall=wall, restart=restart))
    _lf(local_dir)
    _ssh("mkdir -p %s" % remote)
    _scp(local_dir + "/.", "%s:%s" % (HOST, remote))
    _ssh("cd %s && for f in INCAR KPOINTS POSCAR job.slurm; do [ -f $f ] && sed -i 's/\\r$//' $f; done; rm -f STAGE_DONE" % remote)
    jid = _ssh("cd %s && sbatch --parsable job.slurm" % remote).splitlines()[-1].strip()
    for _ in range(max_polls):
        if "Y" in _ssh("test -f %s/STAGE_DONE && echo Y" % remote):
            break
        time.sleep(poll)
    else:
        return None
    for f in fetch:
        _scp("%s:%s/%s" % (HOST, remote, f), os.path.join(local_dir, os.path.basename(f)))
    return jid

def measure_gap(eigenval, nelect):
    # fundamental gap = global CBM - global VBM across all sampled k (correct for INDIRECT-gap
    # materials like Si/diamond/SiC, where the direct Gamma gap is much larger and would bias c* low)
    return tune_cmbj.measure_gap(eigenval, nelect)["indirect"]

def grad_ratio_from_chgcar(chgcar):
    """<|grad rho|/rho> in 1/Angstrom, the physics feature (same integrand as compute_cmbj)."""
    from pymatgen.io.vasp.outputs import Chgcar
    chg = Chgcar.from_file(chgcar); L = np.array(chg.structure.lattice.matrix)
    V = abs(np.linalg.det(L)); rho = np.array(chg.data['total']) / V; NG = rho.shape
    gf = [np.gradient(rho, 1.0/NG[i], axis=i, edge_order=2) for i in range(3)]
    LiT = np.linalg.inv(L).T
    gm = np.sqrt(sum((sum(LiT[a, b]*gf[b] for b in range(3)))**2 for a in range(3)))
    m = rho > 1e-4*rho.mean()
    return float((gm[m]/rho[m]).mean())

def anion_p_center(vasprun):
    """Anion-p band center (eV, rel. E_F) from a PBE DOS run (LORBIT=11): occupied first moment of the
    most-electronegative element's p-projected DOS. Deeper (more negative) => more ionic => higher c*."""
    from pymatgen.io.vasp.outputs import Vasprun
    from pymatgen.electronic_structure.core import OrbitalType
    vr = Vasprun(vasprun, parse_dos=True, parse_eigen=False, parse_potcar_file=False)
    cdos = vr.complete_dos; ef = cdos.efermi
    anion = max(vr.final_structure.composition.elements, key=lambda e: (e.X if e.X else 0))
    pdos = cdos.get_element_spd_dos(anion)[OrbitalType.p]
    E = np.array(pdos.energies) - ef
    rho = sum(np.array(v) for v in pdos.densities.values())
    occ = E <= 0.0
    den = np.trapz(rho[occ], E[occ])
    return float(np.trapz((E*rho)[occ], E[occ]) / den) if den > 0 else float("nan")

# ----------------------------- probe stage (self-contained hybrid mesh) -----------------------------
def build_probe_kpoints(kpath_file, out, divs=(4, 4, 4)):
    """Explicit hybrid KPOINTS: full Gamma mesh (weighted 1, seeds the SCF density) + the coarse
    zero-weight high-symmetry path (gives the gap). No master IBZKPT needed -> one self-contained job."""
    nx, ny, nz = divs
    mesh = ["%.8f %.8f %.8f 1" % (i/nx, j/ny, k/nz) for i in range(nx) for j in range(ny) for k in range(nz)]
    path = [l for l in open(kpath_file).read().splitlines() if l.strip()]
    lines = ["Hybrid: %dx%dx%d weighted mesh + zero-weight path" % (nx, ny, nz),
             str(len(mesh) + len(path)), "Reciprocal"] + mesh + path
    open(out, "w").write("\n".join(lines) + "\n")

# ----------------------------- one material -----------------------------
def process(formula, spacegroup, ref_gap, mp_pbe_gap, workroot, potcar_dir, encut=500):
    name = "%s_%s" % (formula, spacegroup.replace("/", ""))
    d = os.path.join(workroot, name); os.makedirs(d, exist_ok=True)
    poscar = os.path.join(d, "POSCAR")
    import shutil as _sh
    prefetched = os.path.join("structures", name, "POSCAR")   # fetch_all.py: correct ground-state polymorph
    if os.path.exists(prefetched):
        _sh.copy(prefetched, poscar)
    else:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_fetch.py"), "--formula", formula,
                        "--spacegroup", spacegroup, "--out", poscar], check=True)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_pipeline.py"), "--poscar", poscar,
                    "--name", name, "--outdir", d, "--encut", str(encut), "--primitive",
                    "--potcar-dir", potcar_dir, "--pts-per-seg", "12"], check=True)  # primitive + coarse path
    # --- relax (PBEsol) writing CHGCAR for the density gradient feature ---
    rlx = os.path.join(d, "0_relax")
    inc = open(os.path.join(rlx, "INCAR")).read().replace("LCHARG = .FALSE.", "LCHARG = .TRUE.")
    if os.environ.get("MBJ_SINGLEPOINT"):   # skip the fragile ISIF=3 CG relax; single-point on the (MP-relaxed) cell
        import re as _re
        inc = _re.sub(r"IBRION\s*=\s*-?\d+", "IBRION = -1", inc)
        inc = _re.sub(r"NSW\s*=\s*\d+", "NSW = 0", inc)
        inc = _re.sub(r"ISIF\s*=\s*\d+", "ISIF = 2", inc)
    open(os.path.join(rlx, "INCAR"), "w").write(inc)
    if not run_stage(rlx, "%s/relax" % name, fetch=["CONTCAR", "CHGCAR"], wall="04:00:00", jobname=name[:8]+"rlx"):
        raise RuntimeError("relax timed out")
    struct = Structure.from_file(os.path.join(rlx, "CONTCAR"))
    grad   = grad_ratio_from_chgcar(os.path.join(rlx, "CHGCAR"))
    feats  = descriptors(struct, mp_pbe_gap, grad)         # MP PBE gap as the cheap gap feature
    ne     = nelect_of(struct, potcar_dir)
    # --- DOS stage (PBE SCF, LORBIT=11) -> anion-p band_center feature ---
    import shutil as _s2
    dos = os.path.join(d, "dos"); os.makedirs(dos, exist_ok=True)
    open(os.path.join(dos, "INCAR"), "w").write(
        "SYSTEM = anion-p DOS (PBE SCF, LORBIT=11)\nISTART = 0\nICHARG = 2\nLORBIT = 11\n"
        " PREC = Accurate\n ENCUT = %d\n LREAL = .FALSE.\n NCORE = 9\nISMEAR = 0\n SIGMA = 0.10\n"
        " NEDOS = 2001\nIBRION = -1\n   NSW = 0\n  NELM = 100\n EDIFF = 1.0e-05\n LWAVE = .FALSE.\nLCHARG = .FALSE.\n" % encut)
    _s2.copy(os.path.join(rlx, "CONTCAR"), os.path.join(dos, "POSCAR"))
    _s2.copy(os.path.join(d, "2_bands", "POTCAR"), os.path.join(dos, "POTCAR"))
    _s2.copy(os.path.join(rlx, "KPOINTS"), os.path.join(dos, "KPOINTS"))
    if not run_stage(dos, "%s/dos" % name, fetch=["vasprun.xml"], wall="02:00:00", jobname=name[:8]+"dos"):
        raise RuntimeError("dos timed out")
    feats["band_center"] = round(anion_p_center(os.path.join(dos, "vasprun.xml")), 4)
    # --- two mBJ gap-probes at c_lo, c_hi (one self-contained hybrid-mesh job each) ---
    pts = []
    for c in (1.2, 1.7):
        pd = os.path.join(d, "probe_c%.2f" % c); os.makedirs(pd, exist_ok=True)
        tune_cmbj_probe(pd, c, name, encut)                                 # mBJ INCAR (ICHARG=2, fixed c)
        import shutil
        shutil.copy(os.path.join(rlx, "CONTCAR"), os.path.join(pd, "POSCAR"))
        shutil.copy(os.path.join(d, "2_bands", "POTCAR"), os.path.join(pd, "POTCAR"))
        build_probe_kpoints(os.path.join(d, "2_bands", "KPATH_zeroweight.txt"),
                            os.path.join(pd, "KPOINTS"))
        if not run_stage(pd, "%s/probe_c%.2f" % (name, c), fetch=["EIGENVAL"], wall="02:00:00", jobname=name[:8]+"p"):
            raise RuntimeError("probe c=%.2f timed out" % c)
        pts.append((c, measure_gap(os.path.join(pd, "EIGENVAL"), ne)))
    c_star, m, b, r2 = tune_cmbj.fit_c(pts, ref_gap)
    return dict(formula=formula, spacegroup=spacegroup, ref_gap=ref_gap, mp_pbe_gap=mp_pbe_gap,
                c_lo=pts[0][0], gap_lo=round(pts[0][1], 4), c_hi=pts[1][0], gap_hi=round(pts[1][1], 4),
                dGap_dc=round(m, 4), c_star=round(float(c_star), 4), fit_r2=round(r2, 4), **feats)

def nelect_of(struct, potcar_dir):
    from build_pipeline import potsym, zval
    return int(round(sum(zval(potsym(str(sp))) * struct.composition[sp] for sp in struct.composition)))

def tune_cmbj_probe(folder, c, name, encut):
    subprocess.run([sys.executable, "../tune_cmbj.py", "probe", "--outdir", folder,
                    "--c", str(c), "--name", name, "--encut", str(encut)], check=True)

# ----------------------------- main loop -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", required=True, help="CSV: formula,spacegroup,ref_gap")
    ap.add_argument("--out", default="cstar_dataset.csv")
    ap.add_argument("--workroot", default="runs_ml")
    ap.add_argument("--potcar-dir", default="E:/VASP/potentials/potpaw_PBE.54")
    ap.add_argument("--report", default="fetch_report.csv", help="fetch_all.py output (for MP PBE gap)")
    a = ap.parse_args()
    # MP PBE gap per (formula, sg) from the fetch report, if present
    mpg = {}
    if os.path.exists(a.report):
        for r in csv.DictReader(open(a.report)):
            try: mpg[(r["formula"], r["req_sg"])] = float(r.get("mp_pbe_gap") or 0.0)
            except Exception: pass
    rows = []
    for rec in csv.DictReader(open(a.materials)):
        key = (rec["formula"], rec["spacegroup"])
        try:
            rows.append(process(rec["formula"], rec["spacegroup"], float(rec["ref_gap"]),
                                max(mpg.get(key, 0.0), 0.0), a.workroot, a.potcar_dir))
            print("OK  %s %s  c* = %.3f" % (rec["formula"], rec["spacegroup"], rows[-1]["c_star"]))
        except Exception as e:
            print("SKIP %s: %s" % (rec.get("formula"), e))
    if rows:
        keys = list(rows[0].keys())
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print("wrote %s (%d rows)" % (a.out, len(rows)))

if __name__ == "__main__":
    main()
