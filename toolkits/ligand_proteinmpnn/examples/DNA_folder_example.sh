#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_1.out

source activate mlfold

folder_with_pdbs="../PDB_DNA/"
path_for_parsed_chains="../PDB_DNA/parsed_pdbs.jsonl"
path_for_designed_sequences="../PDB_DNA/all_temp_0.1"

python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 10 \
        --sampling_temp "0.1 0.5" \
        --use_DNA_RNA 1 \
        --use_sc 1 \
        --batch_size 1
