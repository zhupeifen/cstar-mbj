#!/usr/bin/env python
"""Compute the TB-mBJ c-parameter from a converged CHGCAR, self-contained:
    c = CMBJA + CMBJB * sqrt( (1/V) integral |grad rho| / rho  d^3r )
with the VASP-default universal constants CMBJA=-0.012, CMBJB=1.023 (bohr^0.5).
The gradient integral is a robust property of the density, so any reasonable
converged CHGCAR (PBE/PBEsol/mBJ) gives essentially the same c. Needed because
the orca vasp.5.4.4 build does NOT compute c self-consistently (stuck at 2.0).

Usage: py compute_cmbj.py CHGCAR [--a -0.012] [--b 1.023] [--floor 1e-4]
"""
import sys, argparse
import numpy as np
from pymatgen.io.vasp.outputs import Chgcar

BOHR = 0.52917721067   # Angstrom / bohr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chgcar")
    ap.add_argument("--a", type=float, default=-0.012)
    ap.add_argument("--b", type=float, default=1.023)
    ap.add_argument("--floor", type=float, default=1e-4, help="rho floor (frac of mean) to skip vacuum")
    args = ap.parse_args()

    chg = Chgcar.from_file(args.chgcar)
    L = np.array(chg.structure.lattice.matrix)      # Angstrom, rows = lattice vectors
    V = abs(np.linalg.det(L))
    rho = np.array(chg.data['total']) / V           # -> e/Angstrom^3 (CHGCAR stores rho*V)
    NG = rho.shape

    # fractional-coordinate gradients (periodic), then transform to Cartesian: grad_cart = L^{-T} grad_frac
    gf = [np.gradient(rho, 1.0/NG[d], axis=d, edge_order=2) for d in range(3)]  # d rho / d f_d
    Linv_T = np.linalg.inv(L).T
    gx = Linv_T[0,0]*gf[0] + Linv_T[0,1]*gf[1] + Linv_T[0,2]*gf[2]
    gy = Linv_T[1,0]*gf[0] + Linv_T[1,1]*gf[1] + Linv_T[1,2]*gf[2]
    gz = Linv_T[2,0]*gf[0] + Linv_T[2,1]*gf[1] + Linv_T[2,2]*gf[2]
    gmag = np.sqrt(gx*gx + gy*gy + gz*gz)           # |grad rho|, e/Angstrom^4

    floor = args.floor * rho.mean()
    mask = rho > floor
    ratio = np.zeros_like(rho)
    ratio[mask] = gmag[mask] / rho[mask]            # 1/Angstrom
    # g = (1/V) integral ratio dV = mean(ratio) over the grid  (dV = V/Npts)
    g_ang = ratio.mean()                            # 1/Angstrom
    g_bohr = g_ang * BOHR                            # 1/bohr
    c = args.a + args.b * np.sqrt(g_bohr)
    print("V=%.3f A^3  grid=%s  <|grad rho|/rho>=%.5f /A  =%.5f /bohr" % (V, NG, g_ang, g_bohr))
    print("CMBJ c = %.4f   (A=%.3f, B=%.3f)" % (c, args.a, args.b))
    return c

if __name__ == "__main__":
    main()
