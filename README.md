# cstar-mbj — machine-learned TB-mBJ exchange parameter for near-experimental band gaps

Predict the **material-specific optimal Tran–Blaha mBJ parameter, c\***, from inexpensive PBE
descriptors, so that a **single meta-GGA calculation** at the predicted c\* yields a
near-experimental band gap. This repository contains the dataset, the trained model, the
prediction tool, and the full automated pipeline behind the paper.

> *Machine-learned, material-specific Tran–Blaha exchange parameter for one-shot
> near-experimental band gaps*, P. Zhu and H. Zhu (2026).

## Why

The TB-mBJ meta-GGA gives near-hybrid band gaps at semilocal cost, but its accuracy depends on
one empirical parameter *c* that is material-dependent and unknown a priori. The built-in
density-based self-consistent *c* is **anti-predictive** for the gap-reproducing value
(leave-one-out R² = −0.74 across our 128-material set). We instead **learn** c\*: a Gaussian-process
model predicts it with a leave-one-out MAE of **0.100** (≈0.3 eV in the gap). A single mBJ run at
the predicted c\* reaches experiment with a mean absolute error of **0.37 eV**, versus 0.56 eV for
the best possible universal *c* and 0.76 eV for the self-consistent prescription.

## What's here

```
cstar-mbj/
├── data/
│   ├── cstar_dataset.csv        # 152 materials (full, with QC labels)
│   ├── cstar_dataset_clean.csv  # 128 clean materials (model training set)
│   └── cstar_dataset_qc.csv     # QC-annotated copy
├── model/
│   ├── cstar_model.joblib       # trained GP (primary) + RF, 15 features
│   └── cstar_model.meta.json    # features, CV scores, training date
├── src/
│   ├── predict_cstar.py         # >>> the deployment tool <<<
│   ├── gen_cstar_dataset.py     # full pipeline: MP → PBEsol → 2 mBJ probes → fit c*
│   ├── train_save_cstar.py      # retrain and save the model
│   ├── fit_cstar_model.py       # CV ladder + feature importance
│   ├── benchmark_fixedc.py      # fixed-c vs material-specific c* (gap space)
│   ├── fig_prep.py, gen_tables.py
│   ├── build_pipeline.py, tune_cmbj.py, compute_cmbj.py, mp_fetch.py  # pipeline internals
└── matlab/
    └── Plot_paper_figs.m        # paper figures
```

## Install

```
pip install -r requirements.txt
```
The saved model was trained with **scikit-learn 1.3.0**; loading it under a very different
version may warn (pin 1.3.x for exact reproduction).

## Use the model (the common case)

Given one cheap PBE relaxation (writing `CONTCAR` + `CHGCAR`) and one PBE density-of-states run
(`LORBIT=11`, for the anion-p band-center feature):

```
python src/predict_cstar.py \
    --contcar CONTCAR --chgcar CHGCAR \
    --dos-vasprun dos/vasprun.xml \
    --pbe-gap 0.61            # or omit and pass --vasprun for auto gap
```
Output:
```
predicted c* = 1.31 +/- 0.05  (GP, primary)  |  1.30  (RF, cross-check)
  -> run one TB-mBJ SCF with CMBJ = 1.31
```
Then run a single mBJ SCF with `CMBJ = <predicted c*>` for a near-experimental gap and a
self-consistent density/band structure.

## Reproduce the model and analysis

The analysis scripts read the dataset CSVs from the working directory:
```
cd src
cp ../data/*.csv .
python train_save_cstar.py        # retrain -> cstar_model.joblib (LOO CV printed)
python fit_cstar_model.py         # CV ladder + feature importances
python benchmark_fixedc.py        # fixed-c vs ML c* (gap-space benchmark)
python fig_prep.py && python gen_tables.py
```

## Regenerate the c\* dataset (requires VASP + a SLURM cluster)

`gen_cstar_dataset.py` drives the full first-principles pipeline (Materials Project fetch →
PBEsol relax → two fixed-c mBJ probes → linear fit → QC). It needs VASP, POTCARs, and a cluster;
see the header of that file and `build_pipeline.py` for configuration.

## Data description

Each row of `cstar_dataset.csv`: formula, space group, experimental gap, PBE gap, the two
probe results, the fitted **c\***, fit quality, the 15 descriptors (composition/electronegativity
statistics, charge-density-gradient ratio, self-consistent-c estimate, PBE gap, and the anion-p
band center), and QC label (`clean` = used for the model).

## License

MIT — see `LICENSE`.

## Citation

See `CITATION.cff`.
