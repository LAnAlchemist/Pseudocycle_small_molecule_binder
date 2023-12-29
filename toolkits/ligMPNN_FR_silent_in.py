####Gyu Rie Lee, 2022
#MPNN part copied and simplified from Justas's ligand_proteinmpnn/protein_mpnn_run.py. (skipping the parts related to native seq (seq recovery,..etc)
#MPNN-FR concept and some format borrowed from Nate B and bcov's PPI pipeline
#This is a quick-dirty version. Need to reformat
#Linna An edit
#1. added some simple functions to fit in LA's design pipeline
#2. remove rely on native pose, just generate CA cst from import pose and apply those before FR each time
#3. added check_point function
#4. allow first round for freeze residues
#5. change repack to all residues 5 A close to ligand
#6. add options to use ligandMPNN, v1 and v2
#7. add options to save ligMPNN scores

import json, time, os, sys, glob
import shutil
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
import os.path

import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.protocols.simple_moves import *
#import pyrosetta.rosetta.protocols.rosetta_scripts as rosetta_scripts

import pyrosetta.distributed.io as io
import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
import pyrosetta.distributed.tasks.score as score
sys.path.append( '/home/linnaan/lib/' )
#sys.path.insert(0,'/home/gyurie/scripts')
#from get_pocket_res_by_dist import PocketPDB
# sys.path.insert(0,'/projects/small-molecule/gyurie/ligand_mpnn_FR')
#GRL ver
#from xml_relax_after_ligMPNN_LAv import XML_BSITE_REPACK_MIN, XML_BSITE_FASTRELAX
from xml_relax_after_ligMPNN_LAver import XML_BSITE_REPACK_MIN, XML_BSITE_FASTRELAX
from gen_prot_lig_dist_cst import extract_dist_cst_from_pdb,CST_STDERR


from libCommonJupyterFunc import get_total_scores

#####TODO: Convert these mpnn-related global variables as locals and split this script.
########

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#parse MPNN related arguments

#file_path = os.path.realpath(__file__)
#k = file_path.rfind("/")
#model_weigths_path = file_path[:k] + '/model_weights/lnet_plus4/epoch870_step231040.pt'
#model_weigths_path = file_path[:k] + '/model_weights/lnet_plus10/epoch2000_step219824.pt'
model_weights_path = '/home/gyurie/mpnn_ligand/bin/proteinmpnn/ligand_proteinmpnn/model_weights/lnet_plus10/epoch2000_step219824.pt'
argparser.add_argument("--silent", type=str, default='', help="input silent file")
argparser.add_argument("--tags", type=str, default='', help="input tags from the silent file, connected by comma")
argparser.add_argument("--checkpoint_path", type=str, default=model_weights_path, help="Path to the model checkpoint")# noise version: /home/gyurie/mpnn_ligand/bin/proteinmpnn/ligand_proteinmpnn/model_weights/lnet_plus10_020/epoch2000_step219852.pt
#argparser.add_argument("--use_noise", type=bool, default=False, help="if true, use noise version of checkpoint")
#argparser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for the model")
#argparser.add_argument("--num_layers", type=int, default=3, help="Number of layers for the model")
argparser.add_argument("--num_connections", type=int, default=48, help="Default 48")
argparser.add_argument("--save_score", type=int, default=0, help="0 for False, 1 for True; save score=-log_prob to npy files")
#argparser.add_argument("--save_probs", type=int, default=0, help="0 for False, 1 for True; save MPNN predicted probabilites per position")
#argparser.add_argument("--score_only", type=int, default=0, help="0 for False, 1 for True; score input backbone-sequence pairs")
argparser.add_argument("--use_sc", type=int, default=1, help="0 for False, 1 for True; use side chain context for fixed residues")
argparser.add_argument("--use_DNA_RNA", type=int, default=0, help="0 for False, 1 for True; use RNA/DNA context")
argparser.add_argument("--use_ligand", type=int, default=1, help="0 for False, 1 for True; use ligand context")
argparser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
###mod: num_seq_per_target will be applied to the first iteration only
argparser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
#batch_size should probably be fixed as 1
#argparser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
argparser.add_argument("--max_length", type=int, default=20000, help="Max sequence length")
argparser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")
argparser.add_argument("--out_folder", type=str, default='ligMPNN_FR_out', help="Path to a folder to output sequences, e.g. /home/out/")
argparser.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed")
argparser.add_argument("--ligand_params_path", type=str, default='', help="Path to a params file for the single PDB")
#argparser.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
#argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl")
#argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
argparser.add_argument("--omit_AAs", type=list, default='CX', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
#argparser.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
#argparser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
#argparser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
#argparser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
#argparser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
#argparser.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")

##Relaxation related arguments.
argparser.add_argument("--ligand_genpot_params_path", type=str, default='', help="Path to a genpot params file. to be used with gen potential")
argparser.add_argument("--use_genpot_relax", type=int, default=1, help="0 for no genpot, 1 for True yes genpot during relax")
argparser.add_argument("--n_mpnn_FR_cycle", type=int, default=1, help="Number of cycles to repeat mpnn and relax")
argparser.add_argument("--repackable_res", type=str, default='', help="repackable residue numbers concatenated with ,")
argparser.add_argument("--target_hb_atms", type=str, default='', help="Target ligand atom names to evaluate hbond to protein")
argparser.add_argument("--dump_pdb", type=int, default=0, help="to write all trace pdbs")
argparser.add_argument("--target_atm_for_cst", type=str, default='', help="Target ligand atom names to extract distance constraints from the input design")
argparser.add_argument("--out_name", type=str, default="", help="out name for checkpointing")
argparser.add_argument("--suffix", type=str, default='', help="add to the end of the pdb, not contain _")
argparser.add_argument("--ligand_res_number", type=int, default=0, help="add ligand_res_number")
argparser.add_argument("--debug_mode", type=int, default=0, help="use 1 to turn debug mode on")
argparser.add_argument("--freeze_res_1st_round", type=str,default='True',help="by default, in the first round, key residues get freeze")
argparser.add_argument("--freeze_res_1st_list", type=str, default='',help="provide list connected with comma")
argparser.add_argument("--save_probs", type=int, default=0, help="0 for False, 1 for True; save MPNN predicted probabilites per position")
argparser.add_argument("--save_ligmpnn_score", type=int, default=1, help="save ligand mpnn score or not")
argparser.add_argument("--use_ligmpnn_version", type=str, default='v1Noise', help="define which version of ligmpnn to use")
argparser.add_argument("--mask_hydrogen", type=int, default=1, help="does not use hydrogen since the training set does not has hydrogen")




args = argparser.parse_args()    
#main(args) 
print('freeze_res_1st_round '+str(args.freeze_res_1st_round))

####rosetta-related things. initialize pyrosetta
#TODO: include holes
if args.use_ligmpnn_version == 'v1Noise':
    print(f'Use ligMPNN v1noise!')
    #sys.path.insert(0,'/home/gyurie/mpnn_ligand/bin/proteinmpnn/ligand_proteinmpnn')
    sys.path.insert(0,'/home/linnaan/software/proteinmpnn/ligand_proteinmpnn') #v1proteinmpnn, trained with single chain
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN
elif args.use_ligmpnn_version == 'v2Noise':
    print(f'Use ligMPNN v2 noise!')
    #sys.path.insert(0,'/home/gyurie/mpnn_ligand/bin/proteinmpnn/ligand_proteinmpnn')
    sys.path.insert(0,'/home/linnaan/software/proteinmpnn/ligand_v2') #v1proteinmpnn, trained with single chain
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN



init_cmd = ['-beta']
if bool(args.use_genpot_relax):
    init_cmd.append('-gen_potential -out:file:write_pdb_parametric_info True')
    init_cmd.append('-extra_res_fa %s'%args.ligand_genpot_params_path)
    #coordinate cst will always be generated from this input pose (not updating)
    #init_cmd.append('-in:file:native %s'%args.pdb_path)
    lig_params = args.ligand_genpot_params_path
else:
    init_cmd.append('-out:file:write_pdb_parametric_info True -extra_res_fa %s'%args.ligand_params_path)
    #init_cmd.append('-in:file:native %s'%args.pdb_path) 
    lig_params = args.ligand_params_path

pyrosetta.init(' '.join(init_cmd))

N_MPNN_FR_ITER = args.n_mpnn_FR_cycle
if args.suffix:
    SUFFIX = args.suffix
if args.freeze_res_1st_round == "True":
    freeze_res_1st_round = True
    print('freeze_res_1st_round '+str(args.freeze_res_1st_round))
    freeze_res_1st_round = True
    freeze_res_1st_list = [int(i) for i in args.freeze_res_1st_list.split(',')]
elif args.freeze_res_1st_round == "False":
    freeze_res_1st_round = False
    print('freeze_res_1st_round '+str(args.freeze_res_1st_round))
    freeze_res_1st_list = []

pssm_coef = None
pssm_bias = None


def get_all_atom_close_csts(pose, ligand_res_number, bb_only=False, sd=1.0):
        """
        get cst for ligand and CA cst
        atm_list - one ligand atms, use for estimate distance
        usage:
        cst_list = get_all_atom_close_csts(pose, bb_only=False, bb_sd=0.5, sc_sd=10.0)

            for cst in cst_list:
                pose.add_constraint(cst)
        """
        # bbs = ["N", "O", "C", "CA", "CB"]
        cst_list = []
        ligand = int(ligand_res_number)

        #generate CA cst
        for resi in range(1, pose.size()):
            for at_i in range(1, pose.residue(resi).natoms() + 1):
                if pose.residue(resi).atom_name(at_i).strip() != "CA": #only do CA, could do CB instead but then need logic for GLY
                    continue    
                else:
                    for resj in range(resi, pose.size()):
                        for at_j in range(1, pose.residue(resj).natoms() + 1):
                            if pose.residue(resj).atom_name(at_j).strip() != "CA": #only do CA, could do CB instead but then need logic for GLY
                                continue
                            else:    
                                cst = ""
                                id_i = pyrosetta.rosetta.core.id.AtomID(at_i, resi)
                                id_j = pyrosetta.rosetta.core.id.AtomID(at_j, resj)
                                i_j_dist = pose.residue(resi).xyz(at_i).distance(pose.residue(resj).xyz(at_j))
                                func = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(i_j_dist, sd)
                                cst = pyrosetta.rosetta.core.scoring.constraints.AtomPairConstraint(id_i, id_j, func)
                                cst_list.append(cst)
        print(f'generated CA cst {len(cst_list)}')
        #generate CA-ligand cst
        for resi in range(1, pose.size()):
            for at_i in range(1, pose.residue(resi).natoms() + 1):
                if pose.residue(resi).atom_name(at_i).strip() != "CA": #only do CA, could do CB instead but then need logic for GLY
                    continue    
                best_dist = 11
                cst = ""
                for at_j in range(1, pose.residue(ligand).natoms() + 1):
                    if "H" in pose.residue(ligand).atom_name(at_j).strip():
                        continue
                    if pose.residue(resi).xyz("CA").distance_squared(pose.residue(ligand).xyz(pose.residue(ligand).atom_name(at_j).strip())) >= 100:
                        continue
                    id_i = pyrosetta.rosetta.core.id.AtomID(at_i, resi)
                    id_j = pyrosetta.rosetta.core.id.AtomID(at_j, ligand)

                    i_j_dist = pose.residue(resi).xyz(at_i).distance(pose.residue(ligand).xyz(at_j))

                    if i_j_dist < best_dist:
                        best_dist = i_j_dist
                        func = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(best_dist, sd)
                        cst = pyrosetta.rosetta.core.scoring.constraints.AtomPairConstraint(id_i, id_j, func)
                # if cst != "":
                        cst_list.append(cst)

        print(f'total {len(cst_list)} cst generated')

        return cst_list

#############

START_NO = 1 #Residue number to start sequence threading.

#NUM_BATCHES = args.num_seq_per_target//args.batch_size
#BATCH_COPIES = args.batch_size
batch_size = 1
BATCH_COPIES = batch_size
#
temperatures = [float(item) for item in args.sampling_temp.split()]
omit_AAs_list = args.omit_AAs
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        
omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

chain_id_dict = None

if os.path.isfile(args.fixed_positions_jsonl):
    with open(args.fixed_positions_jsonl, 'r') as json_file:
        json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
else:
    fixed_positions_dict = None

if os.path.isfile(args.omit_AA_jsonl):
    with open(args.omit_AA_jsonl, 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        omit_AA_dict = json.loads(json_str)
else:
    omit_AA_dict = None
    
if os.path.isfile(args.bias_AA_jsonl):
    with open(args.bias_AA_jsonl, 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        bias_AA_dict = json.loads(json_str)
else:
    bias_AA_dict = None
     
bias_AAs_np = np.zeros(len(alphabet))
if bias_AA_dict:
    for n, AA in enumerate(alphabet):
        if AA in list(bias_AA_dict.keys()):
            bias_AAs_np[n] = bias_AA_dict[AA]

##setting None as default for now
tied_positions_dict=None
pssm_dict=None
pssm_threshold=0.0
pssm_bias_flag=0
##          
            
def init_seq_optimize_model():

    hidden_dim = 128 #default
    num_layers = 3 #default
    num_connections = int(args.num_connections) #48 is v1 #defulat

    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=args.backbone_noise, k_neighbors=num_connections, device=device)
    model.to(device)
    # if args.use_noise:
    #     checkpoint_path = '/home/gyurie/mpnn_ligand/bin/proteinmpnn/ligand_proteinmpnn/model_weights/lnet_plus10_020/epoch2000_step219852.pt'
    #     print(f'Use noise!')
    # else:
    #     checkpoint_path = args.checkpoint_path
    checkpoint_path = args.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

    
def generate_sequences( model , pdb_in, fixed_positions_dict=None, seq_per_target_in=1):
    NUM_BATCHES = seq_per_target_in//batch_size
#    NUM_BATCHES = seq_per_target_in//args.batch_size
    #when seq_per_target_in==1 the above will be zero and not run at all..?
    NUM_BATCHES = max(NUM_BATCHES,1)
    #

    #Moving the pdb parser part here inside the function##
    #All pdb would have the same ligand
    if args.ligand_params_path:
        pdb_dict_list = parse_PDB(pdb_in, {pdb_in: [args.ligand_params_path]})
    else:
        pdb_dict_list = parse_PDB(pdb_in)
    
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
    all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    designed_chain_list = all_chain_list
    fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
    
    pssm_multi = 0.0
    pssm_log_odds_flag = 0

    total_residues = 0
    protein_list = []
    total_step = 0
    # Validation epoch
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        #print('Generating sequences...')
        for ix, protein in enumerate(dataset_valid):
            seq_score = {}
#            score_list = []
#            seq_s = []
#            all_probs_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            Z, Z_m, Z_t, X, X_m, Y, Y_m, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef_all, pssm_bias_all, pssm_log_odds_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict)
            if not args.use_sc:
                X_m = X_m * 0
            if not args.use_DNA_RNA:
                Y_m = Y_m * 0
            if not args.use_ligand:
                Z_m = Z_m * 0
            if args.mask_hydrogen:
                print(f'mask hydrogen!') #Z_t - ligand atom type, Z_m - ligand atom mask, Z - ligand atom coords; Z_t==40 will check if the type is hydrogen
                mask_hydrogen = ~(Z_t == 40)  #1 for not hydrogen, 0 for hydrogen
                Z_m = Z_m*mask_hydrogen
                
            #pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float() #1.0 for true, 0.0 for false
            pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
            name_ = batch_clones[0]['name']

            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    #                    if tied_positions_dict == None:
                    sample_dict = model.sample(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=pssm_multi, pssm_log_odds_flag=bool(pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(pssm_bias_flag))
                    S_sample = sample_dict["S"] 
                    #else:
                    #                                sample_dict = model.tied_sample(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), tied_pos=tied_pos_list_of_lists_list[0])
                    #                            # Compute scores
                    #                                S_sample = sample_dict["S"]
                    log_probs = model(X, X_m, Y, Y_m, Z, Z_m, Z_t, randn_2, S_sample, chain_M*chain_M_pos, chain_encoding_all, residue_idx, mask)
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()
#                    all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                    for b_ix in range(BATCH_COPIES):
#                        masked_chain_length_list = masked_chain_length_list_list[b_ix]
#                        masked_list = masked_list_list[b_ix]
                        #seq_recovery_rate = torch.sum(torch.sum(torch.nn.functional.one_hot(S[b_ix], 21)*torch.nn.functional.one_hot(S_sample[b_ix], 21),axis=-1)*mask_for_loss[b_ix])/torch.sum(mask_for_loss[b_ix])
                        seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                        score = scores[b_ix]
                        seq_score[seq] = score
#                        score_list.append(score)
#                        ###
#                        seq_s.append(seq)
                        #native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                        #if b_ix == 0 and j==0 and temp==temperatures[0]:
                        #    start = 0
                        #    end = 0
                        #    list_of_AAs = []
                        #    for mask_l in masked_chain_length_list:
                        #        end += mask_l
                        #        list_of_AAs.append(native_seq[start:end])
                        #        start = end
                        #native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                        #l0 = 0
                        #for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                        #l0 += mc_length
                        #native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                        #l0 += 1
                        #sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                        #print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
                        #sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                        #print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
                        #native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
                        #f.write('>{}, score={}, fixed_chains={}, designed_chains={}\n{}\n'.format(name_, native_score_print, print_visible_chains, print_masked_chains, native_seq)) #write the native sequence
#                        start = 0
#                        end = 0
#                        list_of_AAs = []
#                        for mask_l in masked_chain_length_list:
#                            end += mask_l
#                            list_of_AAs.append(seq[start:end])
#                            start = end
#                        seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
#                        l0 = 0
#                        for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
#                            l0 += mc_length
#                            seq = seq[:l0] + '/' + seq[l0:]
#                            l0 += 1
#                        score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
#                        seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
#                        f.write('>T={}, sample={}, score={}, seq_recovery={}\n{}\n'.format(temp,b_ix,score_print,seq_rec_print,seq)) #write generated sequence
            if args.save_score:
                score_file = base_folder + '/scores/' + batch_clones[0]['name'] + '.npy'
                np.save(score_file, np.array(score_list, np.float32))
    return seq_score

def generate_hb_filters():
    """
    LA added
    """
    filters,protocols = [],[]
    target_hb_atms = args.target_hb_atms.split(',')
    for atm in target_hb_atms:
        filters.append(f'<SimpleHbondsToAtomFilter name="hb_to_{atm}" n_partners="1" hb_e_cutoff="-0.3" target_atom_name="{atm}" res_num="{args.ligand_res_number}" scorefxn="scorefxn_full" confidence="0"/>')
        protocols.append(f'<Add filter="hb_to_{atm}"/>')
    return '\n'.join(filters),'\n'.join(protocols)

def repack_pose(packable_res, xml_in):
    filters,protocols = generate_hb_filters()
    repack_protocol = xml_in.format(packable_res,filters,protocols)
    return rosetta_scripts.SingleoutputRosettaScriptsTask(repack_protocol)

class ThreadSeqRepack:
    def __init__(self,pdb_fn,param_fn,thread_seq_s,i_iter,repackable_res='',\
                 target_hb_atms='',debug=False,output_dir='dump'):
        self.pdb_fn = pdb_fn.strip()
        self.i_iter = i_iter
        self.tag = '%s_FR%d'%(self.pdb_fn.split('/')[-1].split('.pdb')[0],self.i_iter)
        if SUFFIX and (self.i_iter == (N_MPNN_FR_ITER-1)):
            self.tag = '%s_FR%d%s'%(self.pdb_fn.split('/')[-1].split('.pdb')[0],self.i_iter,SUFFIX)
        self.param_fn = param_fn.strip()
        self.thread_seq_s = thread_seq_s
        #
        self.hb_filter = False
        self.target_hb_atms = []
        if target_hb_atms.strip() != '':
            self.hb_filter = True
            self.target_hb_atms = target_hb_atms.split(',')
        #
        self.repack_all = True
        self.repack_res = []
        if repackable_res.strip() != '':
            self.repack_all = False
            self.repack_res = [int(x) for x in repackable_res.split(',')]
        self.debug = debug
        self.output_dir = output_dir
        #even not debug mode, will write the final relaxed pdbs in this dir.
        os.makedirs(self.output_dir,exist_ok=True)
        return
    
    def prep_pose(self):
        pose_in = pyrosetta.pose_from_pdb(self.pdb_fn)
        self.pose_start = pose_in.clone()
        self.nres = self.pose_start.total_residue() #including ligand
        # LA add pose cst lig to surrounding Ca
        for cst in cst_list:
            pose_in.add_constraint(cst)
        print('added cst')
        return pose_in
    def thread_seq(self,pose_work,seq_in):
        thread = SimpleThreadingMover(seq_in,START_NO)
        #
        thread.apply(pose_work)
        return pose_work

    def bsite_repack_min(self,pose_work,tag):
        #Repack all if not any residue is assigned
        if self.repack_all and len(self.repack_res) > 0:
            for ires in range(1,self.nres):
                self.repack_res.append(ires)
        #
        repackable_res_str = ','.join(['%d'%resno for resno in self.repack_res])
        for cst in cst_list:
            pose_work.add_constraint(cst)
        print('added cst')
        xml_obj = repack_pose(repackable_res_str,XML_BSITE_REPACK_MIN)
        xml_obj.setup()
        pose_work = xml_obj.apply(pose_work)
        out_pdb = '%s/%s.pdb'%(self.output_dir,tag)
        packed_pose.to_pose(pose_work).dump_pdb(f'{out_pdb}')
        return pose_work, out_pdb
#        return pose_work.scores, out_pdb
    def bsite_fastrelax(self,pose_work,tag):
        #Repack all if not any residue is assigned
        if self.repack_all and len(self.repack_res) > 0:
            for ires in range(1,self.nres):
                self.repack_res.append(ires)
        #
        repackable_res_str = ','.join(['%d'%resno for resno in self.repack_res])
        for cst in cst_list:
            pose_work.add_constraint(cst)
        print('start fast relax added cst')
        xml_obj = repack_pose(repackable_res_str,XML_BSITE_FASTRELAX)
        xml_obj.setup()
        pose_work = xml_obj.apply(pose_work)
        print(f'pose_work',pose_work)
        out_pdb = '%s/%s.pdb'%(self.output_dir,tag)
#        pose_work.dump_pdb(f'{out_pdb}')
        packed_pose.to_pose(pose_work).dump_pdb(f'{out_pdb}')
        return pose_work, out_pdb   
    def calc_hb(self,pose_work):
        full_pose = packed_pose.to_pose(pose_work)
        hbond_set = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
        full_pose.update_residue_neighbors()
        pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(full_pose,False,hbond_set)
        #
        lig_res = full_pose.residue(self.nres)
        lig_atm_hb = defaultdict(list)
        for lig_atmName in self.target_hb_atms:
            atm_idx = lig_res.atom_index(lig_atmName)
            atm_id = pyrosetta.rosetta.core.id.AtomID(atm_idx,self.nres)
            found_hbs = hbond_set.atom_hbonds(atm_id)
            #
            if (len(found_hbs) == 0):
                continue
            for hb in found_hbs:
                don_resNo = hb.don_res()
                don_atmName = full_pose.residue(don_resNo).atom_name(hb.don_hatm())
                acc_resNo = hb.acc_res()
                acc_atmName = full_pose.residue(acc_resNo).atom_name(hb.acc_atm())
                #
                hb_atm = {don_resNo:don_atmName,acc_resNo:acc_atmName}
                #
                hb_res_pair = [don_resNo,acc_resNo]
                for i_res,resno in enumerate(hb_res_pair):
                    if self.nres == resno:
                        other_resno = hb_res_pair[1-i_res]
                        #if it is intra lig hb, continue
                        if other_resno == resno:
                            continue
                        #to remove gaps in atom names from pose
                        lig_atm_hb[hb_atm[resno].strip()].append((other_resno,hb_atm[other_resno]))
        #
        hb_sc = {}
        for lig_atmName in self.target_hb_atms:
            if lig_atmName not in list(lig_atm_hb.keys()):
                hb_sc['%s_hbond'%lig_atmName] = 0
            else:
                tmp = []
                for hb in lig_atm_hb[lig_atmName]:
                    if hb not in tmp:
                        tmp.append(hb)
                hb_sc['%s_hbond'%lig_atmName] = len(tmp)
        return hb_sc
    def run(self,mode='relax'):
        pose_init = self.prep_pose()
        #
        df_s = []
        out_pdb_s = []
        out_pose_s = []
        for i_seq,seq in enumerate(self.thread_seq_s):
            pose_work = pose_init.clone()
            pose_work = self.thread_seq(pose_work,seq)
            #
            if self.debug and self.output_dir != None:
                pose_work.dump_pdb('%s/%s_%d_bfr_repack.pdb'%(self.output_dir,self.tag,i_seq))
            #pose_work is packedpose
            if mode=='repack':
                pose_work,out_pdb = self.bsite_repack_min(pose_work,'%s_%d'%(self.tag,i_seq))
            elif mode=='relax':
                pose_work,out_pdb = self.bsite_fastrelax(pose_work,'%s_%d'%(self.tag,i_seq))
            sc = pose_work.scores
            packed_pose.to_pose(pose_work).dump_pdb(f'{out_pdb}')
            out_pdb_s.append(out_pdb)
            #
            sc['tag'] = '%s_%d'%(self.tag,i_seq)
            #
            if self.hb_filter:
                hb_d = self.calc_hb(pose_work)
                sc.update(hb_d)
            #
            #debug
#            print (sc)
            #
            df_s.append(pd.DataFrame.from_records([sc]))
        #
        repacked_df = pd.DataFrame()
        if (len(df_s) > 0):
            repacked_df = pd.concat(df_s,ignore_index=True)
        return repacked_df,out_pdb_s

def main(pdb_in):
    #For a single input with (or w/o - probably will work) a ligand, run MPNN and fast relax iteratively
    #record the score trace in dataframe.
    #Would take most of the available mpnn options
    tot_t0 = time.time()

    print(f'start main!________')
    
    #mpnn model
    model = init_seq_optimize_model()

    #Initial pdb is input from argument. For further iterations, it will be updated
    mpnn_in_pdb_s = [pdb_in]
    #
    df_s = []
    pdbs_to_remove = []
    ligMPNN_info = {}
    #TODO: schedule a ramping ?
    for cycle in range(N_MPNN_FR_ITER):
        print(f'!!! cycle {cycle}')
        hb_atms = ''
        repack_res = args.repackable_res
        n_mpnn_seq = 1 #LA the first round generate multiple sequences as args, after that only generate 1 sequence for 1 input.
        relax_mode='repack'
        if cycle == 0:
            n_mpnn_seq = args.num_seq_per_target
            if freeze_res_1st_round:
                fixed_positions_dict = {os.path.basename(pdb_in).replace('.pdb',''):{'A':freeze_res_1st_list}}
            else:
                fixed_positions_dict = None
            print(fixed_positions_dict)
        #
        #only evaluate hb at the last cycle
        #repack all residue at the last cycle
        elif cycle == N_MPNN_FR_ITER - 1:
            hb_atms = args.target_hb_atms
            repack_res = ''
            relax_mode='relax'
            fixed_positions_dict = None
            print(fixed_positions_dict)
        #
        #This could be 1 if we only want to take the lowest mpnn-score seq to proceed. The sequences are sorted by the score
        n_seq_sample = n_mpnn_seq
            
        #ligand mpnn
        #TODO: only take the lowest mpnn score seq? or use all (for the 1st mpnn)
        out_pdbs_for_next_cycle = []
        ##DEbug
        print ('starting cycle %d ins :'%cycle, mpnn_in_pdb_s, 'n_mpnn_seq %d :'%n_mpnn_seq) 
        ##
        for mpnn_in_pdb in mpnn_in_pdb_s:
            t0 = time.time()
            print ('mpnn for :',mpnn_in_pdb, ' numseq :',n_mpnn_seq)
            seq_score = generate_sequences(model, mpnn_in_pdb, fixed_positions_dict, seq_per_target_in=n_mpnn_seq)
            print ('mpnn out seqs ', seq_score)
            ligMPNN_info[f'cyc{cycle}_seq']=seq_score
            #
            #Sort by score value
            seq_score_sorted = sorted(seq_score.keys(), key=lambda item: item[1])
            #
            t1 = time.time()
            print ('mpnn cycle %d time: %f'%(cycle,t1-t0))
            print ('              seqs: %s'%seq_score_sorted)
            #
            thread_relax = ThreadSeqRepack(mpnn_in_pdb,lig_params,\
                                           seq_score_sorted[:n_seq_sample],\
                                           cycle,\
                                           repackable_res=args.repackable_res,\
                                           target_hb_atms=hb_atms,\
                                           debug=args.dump_pdb,\
                                           output_dir=args.out_folder)
            ###
            cycle_df, out_pdb_s = thread_relax.run(mode=relax_mode)
            df_s.append(cycle_df)
            out_pdbs_for_next_cycle.extend(out_pdb_s)
        fixed_positions_dict = None #only fix residue in the first round

        print(f'end cycle {cycle}')
        #
        #Update input of mpnn with these relaxed to accumulate mpnn-relaxation
        mpnn_in_pdb_s = out_pdbs_for_next_cycle
        print ('Update input after cycle %d :'%cycle, mpnn_in_pdb_s)
        #
        #delete all trace pdbs except for the final relax if args.dump_pdb=false
        if (cycle < N_MPNN_FR_ITER - 1) and not args.dump_pdb:
            pdbs_to_remove.extend(out_pdbs_for_next_cycle)
    
    header = pdb_in.split('/')[-1].split('.pdb')[0]
    sc_df = pd.concat(df_s,ignore_index=True)
    outdir = args.out_folder
    sc_df.to_csv(f'{outdir}/{header}_{SUFFIX}_ligmpnn_fr.csv')
    #
    #delete trace pdbs.
    if not args.dump_pdb and len(pdbs_to_remove) > 0:
        for pdbfn in pdbs_to_remove:
            os.remove(pdbfn)
    #
    
    tot_t1 = time.time()
    print (f'Total time: {tot_t1-tot_t0} sec')

    if args.save_ligmpnn_score:
        for cyc in ligMPNN_info:
            for k,v in ligMPNN_info[cyc].items():
                ligMPNN_info[cyc][k] = float(v)
        with open(f'{outdir}/{header}_{SUFFIX}_ligmpnn_score.json','w') as outfile:
            json.dump(ligMPNN_info,outfile)
    return

def record_checkpoint( pdb, checkpoint_filename ):
    with open( checkpoint_filename, 'a' ) as f:
        f.write( pdb )
        f.write( '\n' )

def determine_finished_structs( checkpoint_filename ):
    done_set = set()
    if not os.path.isfile( checkpoint_filename ): return done_set

    with open( checkpoint_filename, 'r' ) as f:
        for line in f:
            done_set.add( line.strip() )

    return done_set

if __name__ == '__main__':
    #somewhere else where can we do this?
    ####Extract input protein-ligand distance csts
    checkpoint_filename = f"{args.out_name}_check.point"

    if args.debug_mode == 0:
        debug = False
    else:
        debug = True

    debug = True

    finished_structs = determine_finished_structs( checkpoint_filename )

    if args.pdb_path != '':
        pdbs = [args.pdb_path]
        for pdb in pdbs:
            if pdb in finished_structs: continue
            else:
                pose = pyrosetta.pose_from_file(pdb)
                ligand_res_number = args.ligand_res_number

                cst_list = get_all_atom_close_csts(pose, ligand_res_number, bb_only=False, sd=1.0)
                header = pdb.split('/')[-1].split('.pdb')[0]

                if debug: main(pdb)

                else:
                    try: main(pdb)
                    except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )
                    except:
                        seconds = int(time.time() - t0)
                        print( "protocols.jd2.JobDistributor: %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )
            # We are done with one pdb, record that we finished
            record_checkpoint( pdb, checkpoint_filename )

    elif (args.silent != '') and (args.tags != ''):
        silent_in = args.silent
        sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
        sfd_in.read_file(silent_in)
        tags = args.tags.split(',')
        for tag in tags:
            pdb = f'{args.out_folder}{tag}.pdb'
            if pdb in finished_structs: continue
            else:
                pose = Pose()
                sfd_in.get_structure( tag ).fill_pose( pose )
                pose.dump_pdb(pdb)
                ligand_res_number = args.ligand_res_number
                cst_list = get_all_atom_close_csts(pose, ligand_res_number, bb_only=False, sd=1.0)
                print(f'generated {len(cst_list)} cst, ready to start !')
                header = tag
                if debug: 
                    main(pdb)
                else:
                    try: 
                        main(pdb)
                    except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )
                    except:
                        seconds = int(time.time() - t0)
                        print( "protocols.jd2.JobDistributor: %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )
            os.remove(pdb)
                # We are done with one pdb, record that we finished
            record_checkpoint( pdb, checkpoint_filename )


    else:
        print(f'[ERROR] need to input silent and tag, or a single pdb!')


