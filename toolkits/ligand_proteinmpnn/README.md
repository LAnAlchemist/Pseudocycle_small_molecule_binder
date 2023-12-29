# ProteinMPNN

![ProteinMPNN](https://docs.google.com/drawings/d/e/2PACX-1vTm7k_yD-OfOUejDNh91rkeNKOaGrKw1RnmYmXRdkjp7VwK-JHdBgMQY2kOwWBNgbYcDAPFiWKZh_e1/pub?w=892&h=463)

-----------------------------------------------------------------------------------------------------
Example to design with a ligand:
```
#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=ligand_example.out

source activate mlfold

path_to_PDB="/home/gyurie/mpnn_ligand/inf_example/srtRe1657.pdb"
path_to_PDB_params="/home/gyurie/mpnn_ligand/inf_example/SRO.params"
path_for_designed_sequences="../PDB_DNA/no_ligand_temp_0.1"

python ../protein_mpnn_run.py \
        --pdb_path $path_to_PDB \
        --ligand_params_path $path_to_PDB_params \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --use_ligand 1 \
        --batch_size 1
```
-----------------------------------------------------------------------------------------------------
To parse a folder with PDBs use `helper_scripts/parse_multiple_chains.py`, parsing will look for params files with the same name as PDB names, e.g. my_protein.pdb and my_protein.params
```
#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_1.out

source activate mlfold

folder_with_pdbs="../ligand_folder/"
path_for_parsed_chains="../ligand_folder/parsed_pdbs.jsonl"
path_for_designed_sequences="../ligand_folder/temp_0.1"

python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 1 \
        --sampling_temp "0.1" \
        --batch_size 1
```

