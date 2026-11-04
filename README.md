# cstar-mbj: machine-learned TB-mBJ exchange parameter for near-experimental band gaps

Predict the **material-specific optimal Tran–Blaha mBJ parameter, c\***, from inexpensive
descriptors, so that a **single meta-GGA calculation** at the predicted c\* gives a
near-experimental band gap. This repository contains the dataset, the trained model, the
prediction tool, and the automated pipeline behind the paper.

> *Machine-learned, material-specific Tran–Blaha exchange parameter for one-shot
> near-experimental band gaps*, P. Zhu and H. Zhu (2026).

## Summary

The TB-mBJ meta-GGA gives near-hybrid band gaps at semilocal cost, but its accuracy depends on
one empirical parameter *c* whose best value is material-dependent and unknown a priori. The
built-in density-based self-consistent *c* predicts the gap-reproducing value worse than a constant
(R² = −0.23 on our 100 clean materials). We define c\* as the value of *c* that reproduces the
measured fundamental gap and learn it from fourteen inexpensive descriptors. A random forest
predicts c\* with a leave-one-out MAE of **0.097** (R² = 0.54), about 0.3 eV in the gap. A single
mBJ calculation at the predicted c\* reaches experiment with a mean absolute error of **0.30 eV**,
against 0.56 eV for the best universal *c* and 0.62 eV for the self-consistent prescription.

## Version 1.2.0 (November 2026): corrected dataset

This version replaces the earlier dataset. The changes are:

- **Experimental gaps with sources.** Every reference gap is traced to a published measurement:
  the benchmark set of Borlido *et al.* [J. Chem. Theory Comput. 15, 5069 (2019)] as revised by
  Grossmann *et al.* [npj Comput. Mater. 12, 25 (2026)], or the original paper. The DOI is in the
  `gap_source` column. 28 materials without a traceable bulk measurement were removed.
- **Converged calculations only.** Every mBJ probe was checked for convergence (final energy
  change below `EDIFF` before `NELM`). Unconverged probes, and the probes of the first batch whose
  output had not been kept, were recalculated with VASP 6.4.2 (`ALGO=All`, `NELM=500`; then
  `ALGO=Damped`, `TIME=0.1`, `NELM=1000`). The columns `probe_c1.20_source` and
  `probe_c1.70_source` record the origin of each probe gap. 33 materials whose probes could not be
  converged were removed; they are listed in `data/removed_materials.csv`.
- **Structures.** All calculations use the Materials-Project structures without relaxation.
  Earlier versions described them as PBEsol-relaxed, which was incorrect.
- **Model.** Retrained on the 100 clean materials with fourteen distinct descriptors (earlier
  versions passed the PBE gap twice).

The final dataset has 116 materials, of which 100 pass quality control and form the training set.

## Contents

```
cstar-mbj/
├── data/
│   ├── cstar_dataset.csv          # 116 materials, with QC labels, gap sources, probe provenance
│   ├── cstar_dataset_clean.csv    # 100 clean materials (training set)
│   ├── cstar_dataset_qc.csv       # copy of the full table
│   ├── removed_materials.csv      # 61 removed materials and the reason for each
│   ├── linearity_check.csv        # third mBJ calculation for 17 materials (paper, Table S3)
│   ├── convergence/               # convergence audit of the original and recalculated probes
│   └── structures/                # Materials-Project structures used (POSCAR); see README there
├── model/
│   ├── cstar_model.joblib         # random forest (primary) + Gaussian process (uncertainty)
│   └── cstar_model.meta.json      # features, CV scores, training date
├── src/
│   ├── predict_cstar.py           # the deployment tool
│   ├── check_convergence.py       # convergence check for a finished mBJ probe
│   ├── gen_cstar_dataset.py       # pipeline: MP structure, PBEsol density, 2 mBJ probes, c*
│   ├── train_save_cstar.py        # retrain and save the model
│   ├── fit_cstar_model.py         # CV ladder and feature importances
│   ├── benchmark_fixedc.py        # fixed c versus material-specific c* (gap space)
│   ├── fig_prep.py, gen_tables.py
│   └── build_pipeline.py, tune_cmbj.py, compute_cmbj.py, mp_fetch.py   # pipeline internals
└── example/                       # runnable demo (SnS2)
```

## Install

```
pip install -r requirements.txt
```
The saved model was trained with scikit-learn 1.9.1; pin that version for exact reproduction.

## Use the model

Inputs: the structure, the charge density (`CHGCAR`) of one PBEsol self-consistent calculation
on it, one PBE density-of-states calculation (`LORBIT=11`) for the anion-p band center, and the PBE
gap (for example the Materials-Project value):

```
python src/predict_cstar.py \
    --contcar POSCAR --chgcar CHGCAR \
    --dos-vasprun dos/vasprun.xml \
    --pbe-gap 0.61
```
Then run one self-consistent TB-mBJ calculation with `CMBJ` set to the predicted c\*. Check that it
converged (`python src/check_convergence.py <run directory>`): for strongly ionic chlorides and
bromides, halide perovskites, and some barium compounds, the self-consistent mBJ calculation may
fail to converge, and the result should then be discarded.

The Gaussian-process σ ranks predictions by reliability but is too small by about a third; multiply
it by about 1.4 before reading it as a standard deviation.

## Try the example

```
cd example && python run_demo.py
```

## Reproduce the model and analysis

```
cd src
cp ../data/*.csv .
python train_save_cstar.py        # retrain -> cstar_model.joblib (leave-one-out scores printed)
python fit_cstar_model.py         # CV ladder + feature importances
python fig_prep.py && python benchmark_fixedc.py && python gen_tables.py
```

## Regenerate the dataset (requires VASP and a SLURM cluster)

`gen_cstar_dataset.py` drives the first-principles pipeline (Materials-Project structure, PBEsol
self-consistent density, two fixed-*c* mBJ probes at c = 1.2 and 1.7, straight-line solution for
c\*, quality control). Check every probe with `check_convergence.py` before using its gap.

## Data description

Each row of `cstar_dataset.csv`: formula, space group, experimental gap (`ref_gap`) and its source
(`gap_source`, a DOI; `gap_source_type`: `borlido` or `paper`), the Materials-Project PBE gap, the
two probe gaps (`gap_lo` at c = 1.2, `gap_hi` at c = 1.7) and their origin, the slope, c\*, the
descriptors, and the QC label (`clean` = used for the model). The column `pbe_gap` repeats
`mp_pbe_gap` and is not used by the model.

## License

MIT; see `LICENSE`.

## Citation

See `CITATION.cff`.
