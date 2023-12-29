"""
scripts adapted from GRL, Nate
For pseudocycles, bundle pseudocycles with the same length
"""
import os
import mock
import numpy as np
import pickle
from typing import Dict

import sys

# sys.path.insert(0, "/projects/ml/alphafold/alphafold_git/")
sys.path.insert(0,'/home/linnaan/software/alphafold/alphafold/')
# LA: changed /projects/ml/alphafold/alphafold_git/alphafold/common/residue_constants.py gone for no reason

from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
# from alphafold.relax import relax
# from alphafold.relax import utils
import os.path
from jax.lib import xla_bridge

import numpy as np

import jax
import jax.numpy as jnp
from string import ascii_uppercase
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd
import glob
import os.path
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from alphafold.common import confidence

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--fasta_fs', dest='fasta_fs', type=str, help='a list of fasta_fs, separated by comma')
parser.add_argument('--out_dir', dest='out_dir', type=str, help='output dirctory')
parser.add_argument('--min_length', dest='min_length', type=int, help='min length')
parser.add_argument('--max_length', dest='max_length', type=int, help='max length')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1,help='depending on the GPU memory if running GPU,same size proteins can be batched together to make run faster')
parser.add_argument('--num_recycle', dest='num_recycle', type=int, default=4,help='depending on the GPU memory if running GPU')
parser.add_argument('--start_idx', dest='start_idx', type=int, default=0,help='for the sequence in fasta, 0 if starting from the very first sequence; usually native in the MPNN output')
parser.add_argument('--end_idx', dest='end_idx', type=int, default=4,help='final idx for the fasta sequene; e.g. start_idx=1, end_idx=2 would run AF2 on a single MPNN outout sequence')
parser.add_argument('--seed', dest='seed', type=int, default=33241,help='seed for AF2')
parser.add_argument('--params_path', dest='params_path', type=str, default="/projects/ml/alphafold",help='path to AF2 params')
parser.add_argument('--num_models', dest='num_models', type=int, default=5,help='it will run [4, 2, 3, 1, 5][:num_models] these models, 4 is used for compiling jax params')
args = parser.parse_args()


path_to_fasta_files = args.fasta_fs
base_folder = args.out_dir
if base_folder[-1] != "/":
    base_folder = base_folder+"/"

min_length = args.min_length
if args.max_length:
    max_length = args.max_length #we only use pdb with the same length
else:
    max_length = int(min_length) + 5
print(min_length,max_length)
batch_size = args.batch_size
num_recycle = args.num_recycle
start_idx = args.start_idx
end_idx = args.end_idx
seed = args.seed
params_path = args.params_path
num_models = args.num_models

save_npz = False
random_seed = seed #try changing seed
use_amber = False #this does not work for homooligomer > 1


if not os.path.exists(base_folder):
    os.makedirs(base_folder)


# from fasta
def parse_fasta(filename,limit=-1):
  '''function to parse fasta'''
  header = []
  sequence = []
  lines = open(filename, "r")
  for line in lines:
    line = line.rstrip()
    if line[0] == ">":
      if len(header) == limit:
        break
      header.append(line[1:])
      sequence.append([])
    else:
      sequence[-1].append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  return header, sequence

# if path_to_fasta_files[-1] == '/':
#   pass
# else:
#   path_to_fasta_files = path_to_fasta_files + '/'
all_files = path_to_fasta_files.split(',')

new_all_files = []
for name_ in all_files:
    new_all_files.append(name_)

all_sequences = []
all_lengths = []
all_names = []
all_homo_state = []
all_unit_lengths = []
for file_ in new_all_files:
  headers, sequences = parse_fasta(file_)
  print(headers, sequences)
  num_of_chains = sequences[0].count('/')+1
  true_L = len(sequences[0])-(num_of_chains-1)
  print(true_L)
  if min_length <= true_L <= max_length:
    print(true_L)
    for n, seq in enumerate(sequences[start_idx:end_idx]): #DO NOT skip the native sequence from MPNN design, the first fasta sequence
      idx = seq.find('/')
      if idx == -1:
        single_unit_seq = seq
      else:
        single_unit_seq = seq[:idx]
      assert len(single_unit_seq)*num_of_chains == true_L
      all_sequences.append(single_unit_seq)
      all_lengths.append(len(single_unit_seq)*num_of_chains)
      all_names.append(file_[file_.rfind('/')+1:file_.rfind('.')]+f'_{n+1}')
      all_homo_state.append(num_of_chains)
      all_unit_lengths.append(len(single_unit_seq))
print(f'all_lengths {all_lengths}')

crop_size = int(np.max(all_lengths))
max_homo = int(np.max(all_homo_state))

#https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
def tree_stack(trees):
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


def get_confidence_metrics(
    prediction_result: Mapping[str, Any]) -> Mapping[str, Any]:
  """Post processes prediction_result to get confidence metrics."""

  confidence_metrics = {}
  confidence_metrics['plddt'] = confidence.compute_plddt(
      prediction_result['predicted_lddt']['logits'])
  if 'predicted_aligned_error' in prediction_result:
    confidence_metrics.update(confidence.compute_predicted_aligned_error(
        prediction_result['predicted_aligned_error']['logits'],
        prediction_result['predicted_aligned_error']['breaks']))
    confidence_metrics['ptm'] = confidence.predicted_tm_score(
        prediction_result['predicted_aligned_error']['logits'],
        prediction_result['predicted_aligned_error']['breaks'])

  return confidence_metrics

def predict_structure(homo_states, unit_lengths, total_length, seq_length_list, name_list, feature_dict_list, model_params, use_model, random_seed=0):  
  """Predicts structure using AlphaFold for the given sequence."""

  plddts, paes, ptms = [], [], []
  unrelaxed_pdb_lines = []
  relaxed_pdb_lines = []

  chains_list = [] 
  idx_res_list = []
  processed_feature_dict_list = []
  for i, feature_dict in enumerate(feature_dict_list):
      Ls = [unit_lengths[i]]*homo_states[i]
      idx_res = feature_dict['residue_index']
      L_prev = 0
      for L_i in Ls[:-1]:
          idx_res[L_prev+L_i:] += 200
          L_prev += L_i
      chains = list("".join([ascii_uppercase[n]*L for n,L in enumerate(Ls)]))
      feature_dict['residue_index'] = idx_res
      idx_res_list.append(feature_dict['residue_index'])
      chains_list.append(chains)
      processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
      processed_feature_dict_list.append(processed_feature_dict)
  
  batched_dict = tree_stack(processed_feature_dict_list)

  for model_name, params in model_params.items():
    if model_name in use_model:
      prediction_result = runner_vmaped(params, jax.random.PRNGKey(random_seed), batched_dict)
      prediction_result_list = tree_unstack(prediction_result)
      for i, prediction in enumerate(prediction_result_list):
          prediction.update(get_confidence_metrics(prediction))
          unrelaxed_protein = protein.from_prediction(processed_feature_dict_list[i], prediction)
          unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
          plddts.append(prediction['plddt'][:seq_length_list[i]])
          paes.append(prediction['predicted_aligned_error'][:seq_length_list[i],:][:,:seq_length_list[i]])

  model_idx = [4, 2, 3, 1, 5]
  model_idx = model_idx[:num_models]
  out = [{} for _ in name_list]
  for n,r in enumerate(model_idx):
    for k, name in enumerate(name_list):
      j = n*len(name_list)+k
      unrelaxed_pdb_path = base_folder+f'{name}_model_{r}.pdb'    
      with open(unrelaxed_pdb_path, 'w') as f:
        f.write(unrelaxed_pdb_lines[j])
      set_bfactor(unrelaxed_pdb_path, plddts[j], idx_res_list[k], chains_list[k])
      out[k][f"model_{r}"] = {"plddt":plddts[j], "pae":paes[j]}
      average_plddts  = round(float(plddts[j].mean()),2)
      with open(unrelaxed_pdb_path, 'a+') as f:
        f.write(f'model_{r} plddt {average_plddts}')
      print(f"{name} model_{r} plddt {average_plddts}") 
  

  if save_npz:
    for k, name in enumerate(name_list):
      np.savez(base_folder+f'{name}', out[k])
  return out


def set_bfactor(pdb_filename, bfac, idx_res, chains):
  I = open(pdb_filename,"r").readlines()
  O = open(pdb_filename,"w")
  for line in I:
    if line[0:6] == "ATOM  ":
      seq_id = int(line[22:26].strip()) - 1
      seq_id = np.where(idx_res == seq_id)[0][0]
      O.write(f"{line[:21]}{chains[seq_id]}{line[22:60]}{bfac[seq_id]:6.2f}{line[66:]}")
  O.close()

# collect model weights
use_model = {}

if "model_params" not in dir(): model_params = {}
for model_name in ["model_4","model_3","model_5","model_1","model_2"][:num_models]:
  use_model[model_name] = True
  print(f'use model {model_name}')
  if model_name not in model_params:
    model_params[model_name] = data.get_model_haiku_params(model_name=model_name+"_ptm", data_dir=params_path)
    if model_name == "model_4": #compile only model 4 and later load weights for other models
      model_config = config.model_config(model_name+"_ptm")
      model_config.data.common.max_extra_msa = 1 # 5120
      model_config.data.eval.max_msa_clusters = 1 # 512
      model_config.data.eval.num_ensemble = 1
      model_config.data.eval.crop_size = crop_size
      model_config.model.num_recycle = num_recycle
      model_config.data.common.num_recycle = num_recycle
      model_runner_4 = model.RunModel(model_config, model_params[model_name])

model_runner = model_runner_4 #global variable
runner_vmaped = jax.vmap(model_runner.apply, in_axes=(None,None,0))

t0 = time.time()
t0_b = time.time()
name_list = []
feature_dict_list = []
seq_lengths = []
unit_lengths = []
homo_states = []
for i in range(len(all_sequences)):
  query_sequence = all_sequences[i]
  name = all_names[i]
  seq_lengths.append(all_lengths[i])
  unit_lengths.append(all_unit_lengths[i])
  homooligomer = all_homo_state[i]
  homo_states.append(homooligomer)
  msa = [query_sequence]
  deletion_matrix = [[0]*len(query_sequence)]

  if homooligomer == 1:
    msas = [msa]
    deletion_matrices = [deletion_matrix]
    Ln = len(query_sequence)
  else:
    msas = []
    deletion_matrices = []
    Ln = len(query_sequence)
  for o in range(homooligomer):
    L = Ln * o
    R = Ln * (homooligomer-(o+1))
    msas.append(["-"*L+seq+"-"*R for seq in msa])
    deletion_matrices.append([[0]*L+mtx+[0]*R for mtx in deletion_matrix])


  feature_dict = {
        **pipeline.make_sequence_features(sequence=query_sequence*homooligomer,
                                          description="none",
                                          num_res=len(query_sequence)*homooligomer),
        **pipeline.make_msa_features(msas=msas,deletion_matrices=deletion_matrices)}
  
  feature_dict_list.append(feature_dict)
  name_list.append(name)

  #if (i+1) % batch_size == 0 or i == len(all_sequences):
  if (i+1) == len(all_sequences):
    out = predict_structure(homo_states, unit_lengths, crop_size, seq_lengths, name_list, feature_dict_list,
                           model_params=model_params, use_model=use_model, random_seed=random_seed)
    sample_size = len(name_list)
    name_list = []
    feature_dict_list = []
    seq_lengths = []
    unit_lengths = []
    homo_states = []
    dt = round(float(time.time()-t0_b),2)
    print(f'Batch Size: {sample_size}, Time Taken: {dt}')
    t0_b = time.time()
  else:
    print('skipped')

t1 = time.time()
print(f'Finished in {t1-t0} seconds')
