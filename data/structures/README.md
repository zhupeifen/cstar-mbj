# Relaxed structures

PBEsol-relaxed structures (VASP POSCAR format, one `.vasp` file per material, named
`<formula>_<spacegroup>.vasp`) for the c\* dataset. These are the cells on which the two-probe
mBJ gap evaluations were performed.

**Coverage: 135 of the 177 materials.** The remaining 42 were computed on scratch storage that
has since been purged; their relaxed structures are not archived here. Each is fully reproducible
with the released workflow — run `src/gen_cstar_dataset.py` on the material's Materials-Project
entry (formula + space group below), which performs the identical PBEsol relaxation.

Materials not included (regenerable via the workflow):

```
AgI_P6_3mc, CdF2_Fm-3m, Cs2AgBiBr6_Fm-3m, Cs2AgInCl6_Fm-3m, CsGeBr3_R3m, CsI_Pm-3m, CsSnI3_Pm-3m,
Cu2O_Pn-3m, Ga2O3_C2/m, GaS_P6_3/mmc, GaSe_P6_3/mmc, GaTe_C2/m, GeO2_P4_2/mnm, HfO2_P2_1/c,
HfS2_P-3m1, Li2O_Fm-3m, LiGaO2_Pna2_1, LiInS2_Pna2_1, Mg2Ge_Fm-3m, Mg2Si_Fm-3m, Mg2Sn_Fm-3m,
Mg3Sb2_P-3m1, MgF2_P4_2/mnm, MoS2_P6_3/mmc, MoSe2_P6_3/mmc, NiO_Fm-3m, PbF2_Fm-3m, PbI2_P-3m1,
PbSe_Fm-3m, PbTe_Fm-3m, PtS2_P-3m1, SnO2_P4_2/mnm, SnO_P4/nmm, SrCl2_Fm-3m, TiO2_I4_1/amd,
TiO2_P4_2/mnm, ZnF2_P4_2/mnm, ZnGa2O4_Fd-3m, ZnGeAs2_I-42d, ZrO2_P2_1/c, ZrS2_P-3m1, ZrSe2_P-3m1
```
