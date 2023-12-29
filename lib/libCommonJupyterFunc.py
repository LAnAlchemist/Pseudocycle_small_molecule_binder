import os
import sys
import re
import random
import numpy as np
import pandas as pd
import math
import glob
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
from shutil import copyfile
import tempfile
import shutil

def get_total_scores(pdb):     
    '''
    Extract total scores 
    '''
    per_res_score_lines = []
    score_lines = []
    fp = open(pdb,'r')
    start_reading_per_res,start_reading = False,False
    for line in fp:
        if line.startswith('#BEGIN_POSE_ENERGIES_TABLE'):
            start_reading_per_res = True
            continue
        elif (start_reading_per_res ==True) and (line.rstrip() != '') and '#END_POSE_ENERGIES_TABLE' not in line:
            items = line.rstrip().split()
            indi_score_line = []
            for item in items:
                try:
                    indi_score_line.append(float(item))
                except:
                    indi_score_line.append(item)
            per_res_score_lines.append(indi_score_line)
        elif line.startswith('#END_POSE_ENERGIES_TABLE'):
            start_reading_per_res,start_reading = False,True
            continue
        elif (start_reading ==True) and (line.rstrip() != ''):
            #score_lines.append(line.rstrip().split())
            items = line.rstrip().split()
            item_line = []
            for item in items:
                try:
                    item_line.append(float(item))
                except:
                    item_line.append(item)
            score_lines.append(item_line)
    fp.close()

#     rotamer_name = '_'.join(pdb.split('/')[-2].split('_')[-2:])
#     new_name = pdb.split('/')[-1].replace('.pdb',f'_{rotamer_name}.pdb')
    score_lines.append(['description',pdb])
#     score_lines.append(['name',new_name]) #use this one to copy
    df_per_res = pd.DataFrame(per_res_score_lines)
    df_per_res.columns = df_per_res.iloc[0]
    df_sel = df_per_res[df_per_res['label'] == 'pose']
    df_sel = df_sel.reset_index() # reset index to 0, for dataframe merge, only lines with same index can be merged
    df = pd.DataFrame(score_lines,columns=['terms','score'])
    df = df.set_index('terms')
    df = df.transpose()
    df = df.reset_index() # reset index to 0, for dataframe merge, only lines with same index can be merged

    total_df = pd.concat([df,df_sel],axis=1)

    return total_df


def save_scores(pdb_list_to_extract_score,path_to_save_sc,DIVIDINE=1000):
    DIVIDINE = 1000
    for n,pdb in enumerate(pdb_list_to_extract_score):
        if n >= 0:
            design_dir = f'{path_to_save_sc}/'
            start = 0
            for i in range(0,int(len(pdb_list_to_extract_score)/DIVIDINE)+1):
                start_time = time.time()
                end = start + DIVIDINE
                df_list = []
                if end >= len(pdb_list_to_extract_score):
                    scorefiles_list = pdb_list_to_extract_score[start:]
                else:
                    scorefiles_list = pdb_list_to_extract_score[start:end]
                start = end
                for pdb in scorefiles_list:
                    if os.stat(pdb).st_size > 0:
                        indi_df = get_total_scores(pdb)
                        df_list.append(indi_df)
                    else:
                        continue
                df = pd.concat(df_list)
                df.to_csv(f'{design_dir}all_after_predictor_scores_{i}.sc')
                print(f'Used {time.time()-start_time} sec write all_after_predictor_scores_{i}.sc')
            scorefiles = glob.glob(f'{design_dir}all_after_predictor_scores_[0-9]*.sc')
            df_list = []
            for scorefile in scorefiles:
                df = pd.read_csv(scorefile)
                df_list.append(df)
            all_score_df = pd.concat(df_list)
            all_score_df.to_csv(f'{design_dir}all_after_predictor_scores_all.sc')
