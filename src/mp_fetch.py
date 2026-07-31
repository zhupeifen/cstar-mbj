#!/usr/bin/env python
"""Fetch a structure from the Materials Project (new REST API) -> POSCAR.
No pymatgen needed: raw urllib + JSON. Key from ~/.mp_api_key.
Usage:
  mp_fetch.py --formula CsPbBr3 [--list]          # list polymorphs
  mp_fetch.py --formula CsPbBr3 --spacegroup Pm-3m --out POSCAR
  mp_fetch.py --mpid mp-XXXX --out POSCAR
"""
import json, os, sys, argparse, urllib.request, urllib.parse

BASE = "https://api.materialsproject.org"

def key():
    p = os.path.expanduser("~/.mp_api_key")
    return open(p).read().strip()

def q(endpoint, params):
    url = BASE + endpoint + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "X-API-KEY": key(), "accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def summary(formula=None, mpid=None):
    params = {"_fields": "material_id,formula_pretty,symmetry,energy_above_hull,nsites,is_stable",
              "_limit": 50}
    if formula: params["formula"] = formula
    if mpid:    params["material_ids"] = mpid
    return q("/materials/summary/", params)["data"]

def get_structure(mpid):
    d = q("/materials/summary/", {"material_ids": mpid, "_fields": "structure", "_limit": 1})["data"]
    return d[0]["structure"]

def poscar(struct, comment):
    latt = struct["lattice"]["matrix"]
    sites = struct["sites"]
    # group by element preserving first-seen order
    order, groups = [], {}
    for s in sites:
        el = s["species"][0]["element"]
        if el not in groups: groups[el] = []; order.append(el)
        groups[el].append(s["abc"])
    lines = [comment, "1.0"]
    for v in latt: lines.append("  %20.16f %20.16f %20.16f" % tuple(v))
    lines.append("  " + "  ".join("%3s" % e for e in order))
    lines.append("  " + "  ".join("%3d" % len(groups[e]) for e in order))
    lines.append("Direct")
    for e in order:
        for abc in groups[e]:
            lines.append("  %20.16f %20.16f %20.16f  %s" % (abc[0], abc[1], abc[2], e))
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formula"); ap.add_argument("--mpid")
    ap.add_argument("--spacegroup"); ap.add_argument("--out", default="POSCAR")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    rows = summary(formula=a.formula, mpid=a.mpid)
    rows.sort(key=lambda r: (r["energy_above_hull"] if r.get("energy_above_hull") is not None else 9e9))
    if a.list or (not a.mpid and not a.spacegroup):
        print("%-14s %-10s %-8s %-10s %-6s %s" % ("mp-id","formula","SG","E_hull","nsites","stable"))
        for r in rows:
            sym = r.get("symmetry") or {}
            print("%-14s %-10s %-8s %-10.4f %-6d %s" % (
                r["material_id"], r["formula_pretty"], sym.get("symbol","?"),
                r.get("energy_above_hull") or -1, r.get("nsites",0), r.get("is_stable")))
        if a.list: return
    pick = None
    if a.mpid: pick = rows[0]
    elif a.spacegroup:
        for r in rows:
            if ((r.get("symmetry") or {}).get("symbol")) == a.spacegroup: pick = r; break
    if pick is None:
        print("No match; use --list then --mpid or --spacegroup"); sys.exit(1)
    st = get_structure(pick["material_id"])
    txt = poscar(st, "%s %s (%s) from Materials Project" % (
        pick["formula_pretty"], pick["material_id"], (pick.get("symmetry") or {}).get("symbol","")))
    open(a.out, "w").write(txt)
    print("WROTE %s : %s %s SG=%s nsites=%d Ehull=%.4f" % (
        a.out, pick["formula_pretty"], pick["material_id"],
        (pick.get("symmetry") or {}).get("symbol"), pick.get("nsites",0),
        pick.get("energy_above_hull") or -1))

if __name__ == "__main__":
    main()
