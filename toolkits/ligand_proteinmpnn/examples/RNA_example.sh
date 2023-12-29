#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_3.out

source activate mlfold

path_to_PDB="/home/justas/projects/lab_github/DNA_RNA_proteinmpnn/PDB_RNA/3WBM.pdb"
chains_to_design="A B C D"
path_for_designed_sequences="../PDB_RNA/temp_0.1"

python ../protein_mpnn_run.py \
        --pdb_path $path_to_PDB \
        --pdb_path_chains "$chains_to_design" \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 32 \
        --sampling_temp "0.1" \
        --batch_size 8
