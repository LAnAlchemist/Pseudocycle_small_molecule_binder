#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_1.out

source activate mlfold

path_for_parsed_chains="/projects/ml/struc2seq/data_for_complexes/datasets/bbb_DNA_valid.jsonl"
path_for_designed_sequences="../PDB_DNA/valid_temp_0.1"

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 4 \
        --sampling_temp "0.1" \
        --score_only 0 \
        --batch_size 2
