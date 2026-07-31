# Example: predict c\* for SnS₂

A runnable, self-contained demonstration of the one-shot workflow on **SnS₂**
(the layered semiconductor 2H-SnS₂, space group P-3m1).

## Inputs (bundled in `SnS2/`)

All produced by *cheap PBE* calculations, gzipped (pymatgen reads `.gz` directly):

| File | What it is |
|------|------------|
| `CONTCAR` | PBE-relaxed structure |
| `CHGCAR.gz` | PBE charge density (for the ⟨\|∇ρ\|/ρ⟩ gradient-ratio feature) |
| `dos_vasprun.xml.gz` | PBE density-of-states run (`LORBIT=11`, for the anion-p band-center feature) |

## Run it

```
python run_demo.py
```
or call the tool directly:
```
python ../src/predict_cstar.py \
    --contcar SnS2/CONTCAR --chgcar SnS2/CHGCAR.gz \
    --dos-vasprun SnS2/dos_vasprun.xml.gz --pbe-gap 1.56
```

## Expected output

```
material: SnS2   PBE gap = 1.56 eV   grad_ratio = 2.710
predicted c* = 1.172 +/- 0.063  (GP, primary)   |  1.190  (RF, cross-check)
  -> run one TB-mBJ SCF with CMBJ = 1.172
```

The reference (two-probe fit) value for SnS₂ is **c\* = 1.181** (see
`data/cstar_dataset.csv`), so the model's prediction is within 0.01. In practice you would then
run a **single** TB-mBJ SCF with `CMBJ = 1.17`, which yields a near-experimental gap together with
a self-consistent density and band structure.
