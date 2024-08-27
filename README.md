# Pseudocycle_small_molecule_binder
- This script goes along with the paper: De novo design of diverse small molecule binders and sensors using Shape Complementary Pseudocycles
https://www.biorxiv.org/content/10.1101/2023.12.20.572602v1
Twitter: @alchemist_an
# Key references:
0. Pyrosetta: https://www.pyrosetta.org/
1. pseudocycle: https://github.com/LAnAlchemist/Psedocycles_NSMB.git
2. silent tools: git@github.com:bcov77/silent_tools.git
3. AF2: https://github.com/google-deepmind/alphafold.git
4. ProteinMPNN: https://github.com/dauparas/ProteinMPNN.git
5. ligandMPNN: https://github.com/dauparas/LigandMPNN
# Pre-installation requirements
1. Rifgen/RifDock: Klima, J.C., Doyle, L.A., Lee, J.D. et al. Incorporation of sensing modalities into de novo designed fluorescence-activating proteins. Nat Commun 12, 856 (2021). https://doi.org/10.1038/s41467-020-18911-w
2. LigandMPNN: see above (or you can also use the customized version in this repo, they are the same, just minor script change to fit in automatic design instead of only return sequences)
3. ProteinMPNN: https://github.com/dauparas/ProteinMPNN.git
4. AF2: see above
# note
LA made minor change for all above scripts, to make the pipeline easy to use for people who are interested in the pseudocycle method, LA uploaded all the dependent packages for ease of use. However, the users can pull from the original repo and use all abovementioned scripts as it is with simple dependency/arguments modification. These 2 methods will not yield any differences for results.
# How to use this script?
- If you want to use pseudocycle to design binders for your own purpose, clone the repo, follow step-by-step, I provided the stepwise scripts till te end of the first round of design. The second roud of design largely used the similar scripts, so you can copy the jupyter notebooks and repurpose for your own use.
- If you just want particular functions, such as the `predictor`, the `iterative ligandMPNN`, the `pssm-based Rosetta design`, then look through the jupyter notebooks, and you should be able to find the individual links of each script.
- To use the script, clone the repo to your local, replace `home_dir` with the cloned location, you should be ready to go.
- docker is highly encouraged, spec files are provided for docker generation. You can also use conda environment, yaml file is provided.
# environment
- One option is to use conda, it's relatively easy. You can find install yaml files in `environment/`
# steps:
1. generate rotamers to use, follow `1_ligand_conformer_generation_with_rdkit.ipynb`
2. dock to pseudocycles and design using pssm-based rosetta design, follow `2a_first_step_shape_complimentary_screening.ipynb`
3. design using iterative ligandMPNN `2b_first_step_ligMPNN.ipynb`.
4. do your wet, find binder
5. resample based on your hits, generate new sequences based on hit scaffolds, fold them, keep those folded within 3 A, plddt > 90, ptm > 0.65
6. redo docking and design, you can use scripts in step 2-3.
