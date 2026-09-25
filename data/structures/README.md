# Relaxed structures

PBEsol-relaxed structures (VASP POSCAR format, one `.vasp` file per material, named
`<formula>_<spacegroup>.vasp` with any spacegroup `/` removed for filesystem safety; the first
line of each file carries the canonical `formula_spacegroup` label). These are the cells on which
the two-probe mBJ gap evaluations were performed.

**Coverage: all 177 materials** (149 clean + 28 quality-control-flagged). Each `.vasp` file is the
`CONTCAR` from the material's PBEsol relaxation.
