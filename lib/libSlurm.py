#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import shutil
import subprocess
from shutil import copyfile
import tempfile
from collections import defaultdict

from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import roc_auc_score
sns.set_context('notebook')  

# Find relevant features  
# Features to draw ROC for, you should have true and predict value for all of them 

class SlurmPrep:
    def __init__(self,script_name,cmd,que_name='short',core_num=1,\
                 slot_num=1,mem_num=4,log_name='que.log'):
        self.script_name = script_name
        self.cmd = cmd
        self.que_name = que_name #'-p'
        self.core_num = core_num #'-n'
        self.slot_num = slot_num #'-N'
        self.memory = mem_num    #'--mem'
        self.log_name = log_name #'-o'
    def make_script(self):
        cont = []
        cont.append('#!/bin/bash')
        cont.append('#SBATCH -p %s'%self.que_name)
        cont.append('#SBATCH -n %d'%self.core_num)
        cont.append('#SBATCH -N %d'%self.slot_num)
        cont.append('#SBATCH --mem=%dg'%self.memory)
        cont.append('#SBATCH -o %s'%self.log_name)
        cont.append('%s'%self.cmd)
        fout = open('%s'%self.script_name,'w')
        fout.write('\n'.join(cont))
        fout.close()
        return
    def submit(self,hold=False):
        #Assuming that I'm in that directory
        if hold:
            os.system('sbatch -H %s'%self.script_name)
        else:
            os.system('sbatch %s'%self.script_name)
        return

def submit_single_file(cmds='commands', submitfile='submit.sh', queue='short', logsfolder='logs', cpus=1, mem='1g',needs_gpu=False,gpu_type='titan'):
    os.makedirs(logsfolder, exist_ok=True)
    if not needs_gpu:
        substr = \
    """#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=linnaan
#SBATCH -p {0}
#SBATCH -c {4}
#SBATCH -N 1
#SBATCH --mem={3}
#SBATCH -o {2}/out.%a
#SBATCH -e {2}/out.%a

CMD=$( {1} )
echo "${{CMD}}" | bash
"""
        substr = substr.format(queue, cmds, logsfolder, cpus, mem)
        with open(submitfile,'w') as f_out:
            f_out.write(substr)
            
    elif needs_gpu:
        substr = \
        """#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=linnaan
#SBATCH -p gpu
#SBATCH --gres=gpu:{4}:1
#SBATCH -c {2}
#SBATCH --mem={3}
#SBATCH -o {1}/out.%a
#SBATCH -e {1}/out.%a

#source activate /software/conda/envs/tensorflow
CMD=$( {0} )
echo "${{CMD}}" | bash
"""
        substr = substr.format(cmds, logsfolder, cpus, mem,gpu_type)
        with open(submitfile,'w') as f_out:
            f_out.write(substr)

def make_submit_file(cmds='commands', submitfile='submit.sh', time='3:00:00', group_size=1, logsfolder='logs', cpus=1, mem='1g',limit_job_to100 = False, needs_gpu=False, gpu_type='titan',conda_env='mlfold'):
    if os.path.dirname(cmds) == '':
        logsfolder = os.path.dirname(cmds)+f'{logsfolder}'
    elif os.path.dirname(cmds)[-1] == '/':
        logsfolder = os.path.dirname(cmds)+f'{logsfolder}'
    else:
        logsfolder = os.path.dirname(cmds)+f'/{logsfolder}'
    os.makedirs(logsfolder, exist_ok=True)
    n_jobs = sum(1 for line in open(cmds))
    groups = int(np.ceil(float(n_jobs)/group_size))
    
    if not needs_gpu:
        substr = \
    """#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=linnaan
#SBATCH -t {0}
#SBATCH -c {5}
#SBATCH -N 1
#SBATCH --mem={6}
#SBATCH -a 1-{1}{7}
#SBATCH -o {4}/out.%a
#SBATCH -e {4}/out.%a

GROUP_SIZE={3}

for I in $(seq 1 $GROUP_SIZE)
do
    J=$(($SLURM_ARRAY_TASK_ID * $GROUP_SIZE + $I - $GROUP_SIZE))
    CMD=$(sed -n "${{J}}p" {2} )
    echo "${{CMD}}" | bash
done
"""
        if limit_job_to100:
            substr = substr.format(time, groups, cmds, group_size, logsfolder, cpus, mem,'%100')
        elif limit_job_to100 == False:
            substr = substr.format(time, groups, cmds, group_size, logsfolder, cpus, mem,'')

        with open(submitfile,'w') as f_out:
            f_out.write(substr)
            
    elif needs_gpu:
        substr = \
        """#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=linnaan
#SBATCH -p gpu
#SBATCH --gres=gpu:{7}:1
#SBATCH -t {6}
#SBATCH -c {3}
#SBATCH -a 1-{5}
#SBATCH --mem={4}
#SBATCH -o {2}/out.%a
#SBATCH -e {2}/out.%a

GROUP_SIZE={1}

#source activate /software/conda/envs/tensorflow
#source activate mlfold
source activate {8}

for I in $(seq 1 $GROUP_SIZE)
do
    J=$(($SLURM_ARRAY_TASK_ID * $GROUP_SIZE + $I - $GROUP_SIZE))
    CMD=$(sed -n "${{J}}p" {0} )
    echo "${{CMD}}" | bash
done
"""
        substr = substr.format(cmds, group_size, logsfolder, cpus, mem, groups,time,gpu_type,conda_env)
        with open(submitfile,'w') as f_out:
            f_out.write(substr)

def make_dist_plots(df, relevant_features,ncols=3):
    nrows = math.ceil(len(relevant_features) / ncols)
    (fig, axs) = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=[15,3*nrows]
    )
    axs = axs.reshape(-1)

    for (i, metric) in enumerate(relevant_features):
#         is_int_arr = np.array_equal(df[metric], df[metric].dropna().astype(int))
#         if is_int_arr:
#             c = Counter(df[metric])
#             sns.barplot(x=list(c.keys()), y=list(c.values()), ax=axs[i], color='grey')
#             axs[i].set_xlabel(metric)
#         else:
            sns.histplot(df[metric].dropna(), ax=axs[i],linewidth=0, kde=True,alpha=0.38)

    sns.despine()
    plt.tight_layout()
    plt.show()

def plot2pandas(df1,df2,features,legend1='sel',legend2='all',ncols=3):
    nrows = math.ceil(len(features) / ncols)
    (fig, axs) = plt.subplots(ncols=ncols, nrows=nrows, figsize=[4*ncols,3*nrows])
    axs = axs.reshape(-1)
    for (i, metric) in enumerate(features):
        sns.histplot(df1[metric].dropna(), ax=axs[i],label=legend1,linewidth=0, kde=True,alpha=0.38,color='blue').legend()
        sns.histplot(df2[metric].dropna(), ax=axs[i],label=legend2,linewidth=0, kde=True,alpha=0.38,color='orange').legend()
    sns.despine()
    plt.tight_layout()
    plt.show()

def get_flags(rifGenFile,add_path=False):
    returnLines = ''
    file_dir = '/'.join(rifGenFile.split('/')[:-1])
    with open(rifGenFile) as f_open:
        start_reading = False
        for line in f_open:
            if 'what you need for docking' in line:
                start_reading = True
                continue
            if '############' in line:
                continue
            if start_reading:
                if add_path:
                    returnLines += line.replace('output',file_dir+'/output')
                else:
                    returnLines += line
    return returnLines

def grep(file, pattern):
    hits = []
    regx = re.compile(pattern)
    try:
        with open(file,'r') as fin:
            for line in fin:
                is_score = re.findall(regx, line)
                if is_score:
                    hits.append(line.strip())
        return hits
    except:
        return False
def sed_inplace(filename, pattern, repl):
    '''
    Perform the pure-Python equivalent of in-place `sed` substitution: e.g.,
    `sed -i -e 's/'${pattern}'/'${repl}' "${filename}"`.
    '''
    # For efficiency, precompile the passed regular expression.
    pattern_compiled = re.compile(pattern)

    # For portability, NamedTemporaryFile() defaults to mode "w+b" (i.e., binary
    # writing with updating). This is usually a good thing. In this case,
    # however, binary writing imposes non-trivial encoding constraints trivially
    # resolved by switching to text writing. Let's do that.
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        with open(filename) as src_file:
            for line in src_file:
                tmp_file.write(pattern_compiled.sub(repl, line))

    # Overwrite the original file with the munged temporary file in a
    # manner preserving file attributes (e.g., permissions).
    shutil.copystat(filename, tmp_file.name)
    shutil.move(tmp_file.name, filename)
def line_prepender(filename, line, strip=True):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        if strip:
            f.write(line.rstrip('\r\n') + '\n' + content)
        else:
            f.write(line + content)

def plot_ROC(df_feature,features,higher_better,real_maker,test_maker):
    # Prep axes; draw figure
    if len(features)==1:
        ncols=2
    else:
        ncols = 4  
    nrows = math.ceil(len(features)*2 / ncols) #because we will also be plotting ROCs  
    (fig, axs) = plt.subplots(  
        ncols=ncols, nrows=nrows, figsize=[4*ncols,3*nrows]  
    )  
    axs = axs.reshape(-1)  

    # Plot  
    for i,f in enumerate(features):  
        #-------- First we plot the scatter ---------  
        # f+'_min': predict value; f+'_full': true value 
        sns.scatterplot(x=df_feature[f+test_maker], y=df_feature[f+real_maker], ax=axs[i*2])    
        #-------- Next we make roc plots ------------  
        ps = [99,95,85,70,50,25,1] if f in higher_better else [5,15,30,50,75,100]
        #ps = [1] if f in higher_better else [5,15,30,50,75,100]  
        cutoffs = np.percentile(df_feature[f+real_maker], ps) 
        print(cutoffs)
        dftmp = pd.DataFrame.from_dict({'percentiles':ps, 'cutoffs': cutoffs}).drop_duplicates(subset='cutoffs') #remove duplicates in case after some percentile, the cutoffs are the same 
        ps = list(dftmp['percentiles'])  
        cutoffs = list(dftmp['cutoffs'])  
        alt_cuts = list(set(df_feature[f+real_maker])) 
        print(len(alt_cuts))
        if len(alt_cuts)<=5: # if actual data number less or equal to cutoff number 
            cutoffs = alt_cuts  
            ps = len(cutoffs)*['_']            
        for percentile, cutoff in zip(ps, cutoffs):  
            print(percentile, cutoff)
            truth = (df_feature[f+real_maker] >= cutoff) if f in higher_better else (df_feature[f+real_maker] <= cutoff)  
            if sum(truth)==0 or sum(truth)==len(truth): # if no training set in range 
                continue  
            pred = df_feature[f+test_maker] if f in higher_better else - df_feature[f+test_maker]  
            print(len(truth), len(pred))
            y_true = truth
            y_score = pred
            AUC = roc_auc_score(y_true, y_score)
            print(AUC)
            fpr, tpr, _ = roc_curve(truth, pred) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html; fpr: false possitive rate; tpr; true possitive rate 
    #         print(fpr, tpr)
            sns.lineplot(fpr,tpr,ax=axs[i*2+1], label=f'{percentile}_c={np.round(cutoff,0)} AUC={np.round(AUC,2)}')#   
            axs[i*2+1].set_xlabel('tpr')  
            axs[i*2+1].set_ylabel('fpr')  
            axs[i*2+1].set_title(f'{f}{test_maker}')  

    plt.tight_layout()  

def plot_df_hits(df1,df2,features,legend1='sel',legend2='previous_hit',ncols=3):
    nrows = math.ceil(len(features) / ncols)
    (fig, axs) = plt.subplots(ncols=ncols, nrows=nrows, figsize=[4*ncols,3*nrows])
    axs = axs.reshape(-1)
    for (i, metric) in enumerate(features):
        sns.histplot(df1[metric].dropna(), ax=axs[i],label=legend1,linewidth=0, kde=True,alpha=0.38,color='blue').legend()
        for l,(j,row) in enumerate(df2.iterrows()):
            axs[i].axvline(x=row[metric],color='green',label=legend2,linestyle='--')
        #sns.histplot(df2[metric].dropna(), ax=axs[i],label=legend2,linewidth=0, kde=True,alpha=0.38,color='orange').legend()
    sns.despine()

    plt.tight_layout()
    plt.show()

