#!/usr/bin/env python
"""check_convergence.py -- decide whether a finished VASP self-consistent run (e.g. an mBJ probe) converged.

A run counts as converged only if its last electronic step has |dE| < EDIFF and the number of
electronic steps is below NELM. A job that merely finishes (Slurm COMPLETED, STAGE_DONE written)
is not enough: VASP stops quietly when it reaches NELM.

Usage:  python check_convergence.py RUNDIR [RUNDIR ...]
"""
import os, re, sys

def check(d):
    inc = open(os.path.join(d, "INCAR")).read()
    nelm = int(re.search(r"NELM\s*=\s*(\d+)", inc).group(1)) if re.search(r"NELM\s*=", inc) else 60
    ediff = float(re.search(r"EDIFF\s*=\s*([0-9.eE+-]+)", inc).group(1)) if re.search(r"EDIFF\s*=", inc) else 1e-4
    steps = [l for l in open(os.path.join(d, "OSZICAR")) if re.match(r"^[A-Z]{3}:", l)]
    if not steps:
        return False, "no electronic steps in OSZICAR"
    de = abs(float(steps[-1].split()[3]))
    ok = len(steps) < nelm and de < ediff
    return ok, f"{len(steps)} steps (NELM={nelm}), last |dE| = {de:.2e} (EDIFF={ediff:.0e})"

if __name__ == "__main__":
    bad = 0
    for d in sys.argv[1:] or ["."]:
        ok, msg = check(d)
        bad += not ok
        print(f"{'CONVERGED    ' if ok else 'NOT CONVERGED'}  {d}: {msg}")
    sys.exit(1 if bad else 0)
