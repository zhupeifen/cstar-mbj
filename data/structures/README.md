# Structures

Materials-Project structures used for every calculation in this work (VASP POSCAR format, one
`.vasp` file per material, named `<formula>_<spacegroup>.vasp` with any spacegroup `/` removed for
filesystem safety; the first line of each file carries the `formula_spacegroup` label). For each
material this is the lowest-energy Materials-Project polymorph at the given composition and space
group, reduced to the primitive cell and used without relaxation. Earlier versions of this folder
described these files as PBEsol-relaxed; that description was incorrect.

The folder covers all 177 materials of the original list. The 116 materials of the final dataset
are those in `../cstar_dataset.csv`; the others are listed, with the reason for removal, in
`../removed_materials.csv`.
