#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_6.out

source activate mlfold

folder_with_pdbs="../PDB_homooligomers/pdbs/"
path_for_parsed_chains="../PDB_homooligomers/parsed_pdbs.jsonl"
path_for_tied_positions="../PDB_homooligomers/tied_pdbs.jsonl"
path_for_designed_sequences="../PDB_homooligomers/temp_0.1"

python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../helper_scripts/make_tied_positions_dict.py --input_path=$path_for_parsed_chains --output_path=$path_for_tied_positions --homooligomer 1

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --tied_positions_jsonl $path_for_tied_positions \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 16 \
        --sampling_temp "0.2" \
        --batch_size 2
