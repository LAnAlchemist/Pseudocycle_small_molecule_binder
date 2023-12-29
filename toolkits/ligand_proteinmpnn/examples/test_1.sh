#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_1.out

source activate mlfold

folder_with_pdbs="/home/acourbet/final_designs/hallucinated/machine_components/eblock_order_03022022/C18_nmers/for_justas_R3_R5/"
path_for_parsed_chains="../PDB_monomers/test_parsed_pdbs.jsonl"
path_for_designed_sequences="../PDB_monomers/v_score"

python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $path_for_designed_sequences \
        --num_seq_per_target 100 \
        --sampling_temp "0.1" \
        --score_only 1 \
        --batch_size 4
