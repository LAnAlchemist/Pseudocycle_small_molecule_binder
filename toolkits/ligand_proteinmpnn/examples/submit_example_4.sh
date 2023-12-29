#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_4.out

source activate mlfold

folder_with_pdbs="../PDB_complexes/pdbs/"
path_for_parsed_chains="../PDB_complexes/parsed_pdbs.jsonl"
path_for_assigned_chains="../PDB_complexes/assigned_pdbs.jsonl"
path_for_fixed_positions="../PDB_complexes/fixed_pdbs.jsonl"
chains_to_design="A C"
fixed_positions="1 2 3 4 5, 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 40"
path_for_designed_sequences="../PDB_complexes/temp_0.1"

python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"

python ../helper_scripts/make_fixed_positions_dict.py --input_path=$path_for_parsed_chains --output_path=$path_for_fixed_positions --chain_list "$chains_to_design" --position_list "$fixed_positions"

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --chain_id_jsonl $path_for_assigned_chains \
        --fixed_positions_jsonl $path_for_fixed_positions \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 16 \
        --sampling_temp "0.1" \
        --batch_size 8
