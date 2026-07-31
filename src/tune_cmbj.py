#!/usr/bin/env python
"""
tune_cmbj.py -- tune the TB-mBJ c-parameter to reproduce a TARGET (experimental) band gap.

The mBJ gap increases monotonically and ~linearly with c, so two or more (c, gap) data
points let us solve for the c that yields a target gap (a very common published practice;
the self-consistent c does not always match experiment -- see METAGGA wiki / arXiv:2003.10177).

Modes
-----
  gap   --eigenval "BSTR1 EIGENVAL" --nelect 44
            measure the (indirect and min-direct) gap from a VASP EIGENVAL, using the
            zero-weight band-path points if present (else all k-points).

  fit   --points "2.0:3.60,1.478:2.50" --target 2.30
            linear least-squares fit gap(c)=m*c+b through the points, then solve c for the
            target gap. Prints the tuned c and the fit (slope dGap/dc, R^2).

  probe --outdir DIR --c 1.4 [--kpath-file KPATH_zeroweight.txt]
            write a gap-probe INCAR (master mBJ SCF, fixed CMBJ=c) + note. Run it at two c
            values, measure with `gap`, then `fit` -> tuned c. (Orchestration is done by the
            caller / pipeline driver.)

Typical workflow (2 HPC runs):
  1) run the mBJ master+band-edge at c_lo and c_hi (e.g. 1.2 and 1.7)
  2) tune_cmbj.py gap  --eigenval <lo EIGENVAL> --nelect N   -> g_lo
     tune_cmbj.py gap  --eigenval <hi EIGENVAL> --nelect N   -> g_hi
  3) tune_cmbj.py fit  --points "1.2:g_lo,1.7:g_hi" --target G_exp  -> c*
  4) run the full chain with CMBJ=c*
"""
import sys, argparse
import numpy as np

def read_eigenval(path):
    L = open(path, encoding="utf-8", errors="replace").read().split("\n")
    h = [float(x) for x in L[5].split()]; nk = int(h[1]); nb = int(h[2])
    i = 7; kpts = []; E = []
    for k in range(nk):
        while L[i].strip() == "": i += 1
        kpts.append([float(x) for x in L[i].split()]); i += 1
        eb = []
        for b in range(nb):
            eb.append(float(L[i].split()[1])); i += 1
        E.append(eb)
    return np.array(E), np.array(kpts)   # E[nk, nb], kpts[nk, 4]

def measure_gap(path, nelect):
    E, kp = read_eigenval(path)
    zw = kp[:, 3] == 0
    if zw.sum() == 0: zw = np.ones(len(kp), bool)
    Ez = E[zw]; nocc = int(round(nelect / 2))
    vb = Ez[:, nocc - 1]; cb = Ez[:, nocc]
    indirect = cb.min() - vb.max()
    direct = float(np.min(cb - vb)); kd = int(np.argmin(cb - vb))
    return dict(indirect=indirect, direct=direct, k_direct=kd,
                vbm=float(vb.max()), cbm=float(cb.min()), nk=int(zw.sum()))

def fit_c(points, target):
    c = np.array([p[0] for p in points]); g = np.array([p[1] for p in points])
    if len(points) == 1:
        raise SystemExit("need >=2 points to fit gap(c)")
    A = np.vstack([c, np.ones_like(c)]).T
    (m, b), *_ = np.linalg.lstsq(A, g, rcond=None)
    gpred = m * c + b
    ss_res = np.sum((g - gpred) ** 2); ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    c_target = (target - b) / m
    return c_target, m, b, r2

PROBE_INCAR = """SYSTEM = {name} mBJ gap-probe (fixed CMBJ={c})
ISTART = 0
  PREC = Accurate
 ENCUT = {encut}
 LREAL = .FALSE.
 NCORE = 9
METAGGA = MBJ
  CMBJ = {c}
 LASPH = .TRUE.
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
 LWAVE = .FALSE.
LCHARG = .FALSE.
"""

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gap"); g.add_argument("--eigenval", required=True); g.add_argument("--nelect", type=int, required=True)
    f = sub.add_parser("fit"); f.add_argument("--points", required=True); f.add_argument("--target", type=float, required=True)
    p = sub.add_parser("probe"); p.add_argument("--outdir", required=True); p.add_argument("--c", type=float, required=True)
    p.add_argument("--name", default="probe"); p.add_argument("--encut", type=int, default=500)
    a = ap.parse_args()

    if a.mode == "gap":
        r = measure_gap(a.eigenval, a.nelect)
        print("gap: indirect=%.4f  min-direct=%.4f eV  (VBM=%.3f CBM=%.3f, %d k-pts)"
              % (r["indirect"], r["direct"], r["vbm"], r["cbm"], r["nk"]))
    elif a.mode == "fit":
        pts = [tuple(float(x) for x in seg.split(":")) for seg in a.points.split(",")]
        c_t, m, b, r2 = fit_c(pts, a.target)
        print("points (c,gap): %s" % pts)
        print("fit: gap = %.4f*c + %.4f   (dGap/dc=%.3f eV, R^2=%.4f)" % (m, b, m, r2))
        print(">>> tuned CMBJ c = %.4f  to hit target gap %.3f eV" % (c_t, a.target))
    elif a.mode == "probe":
        import os
        os.makedirs(a.outdir, exist_ok=True)
        open(os.path.join(a.outdir, "INCAR"), "w").write(
            PROBE_INCAR.format(name=a.name, c=a.c, encut=a.encut))
        print("wrote %s/INCAR (mBJ gap-probe, CMBJ=%.3f)" % (a.outdir, a.c))

if __name__ == "__main__":
    main()
