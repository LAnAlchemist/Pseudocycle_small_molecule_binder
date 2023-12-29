#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_2.out

source activate mlfold

folder_with_pdbs="../PDB_complexes/pdbs/"
path_for_parsed_chains="../PDB_complexes/parsed_pdbs.jsonl"
path_for_assigned_chains="../PDB_complexes/assigned_pdbs.jsonl"
chains_to_design="A B"
path_for_designed_sequences="../PDB_complexes/temp_0.1"


python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --chain_id_jsonl $path_for_assigned_chains \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 16 \
        --sampling_temp "0.1" \
        --score_only 0 \
        --batch_size 8
