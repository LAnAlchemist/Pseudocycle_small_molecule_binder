import os
import sys
import numpy as np
import math
sys.path.insert(0,'/home/linnaan/lib/')
from libProtein import three_aa_to_one_aa
from collections import defaultdict

class Atom:
    def __init__(self,line):
        self.raw_line = line
        self.line = line.strip()
        self.chainID = line[21]
        self.atmNo = int(line[7:11])
        self.atmName = line[11:17].strip()
        self.resNo = int(line[22:26])
        self.alt_state = line[16]
        self.resName = line[17:21].strip()
        self.oneLetter = three_aa_to_one_aa(self.resName)
        self.coord = [float(line[30:38]),\
                      float(line[38:46]),\
                      float(line[46:54])]
        #
        self.rep_resNo = '%s%d'%(self.chainID,self.resNo)
        self.template = '%s'%(line[:30])
        self.full_template = line.strip()
        return

def dist_two_coords(crd_a,crd_b):
    a = np.array(crd_a)
    b = np.array(crd_b)
    d_xyz = a-b
    d2_sum = []
    for d in d_xyz:
        d2_sum.append(d*d)
    return math.sqrt(sum(d2_sum))


#example
#'AtomPair NE1 11 O 7 HARMONIC 3.0 0.5'
CST_STDERR = 0.5

#Assume there is one type of hetatm ligand and that is our target
def extract_dist_cst_from_pdb(pdb_in,lig_tr_atms,bsite_res=''):
    #
    #use all_res
    if bsite_res == '':
        bsite_res_s = []
        with open(pdb_in) as fp:
            for line in fp:
                if line.startswith('ATOM') and line[13:15] == 'CA':
                    bsite_res_s.append(int(line[22:26]))
    else:
        bsite_res_s = [int(x) for x in bsite_res.split(',')]                    
        
    #Will get distances between each ligand target atom and the closest binding site residue CA
    #First read ligand target atom and binding site residue CA coords
    bsite_CA_R = {}
    het_atm_R = {}
    het_resno = None
    with open(pdb_in) as fp:
        for line in fp:
            if line.startswith('ATOM'):
                atm = Atom(line.strip())
                if atm.resNo in bsite_res_s and atm.atmName == 'CA':
                    bsite_CA_R[atm.resNo] = atm.coord
            elif line.startswith('HETATM'):
                atm = Atom(line.strip())
                het_resno = atm.resNo
                if atm.atmName in lig_tr_atms:
                    het_atm_R[atm.atmName] = atm.coord
    #
    #for each ligand target atm get distances from each bsite residue CA and sort
    cst_s = []
    for hetatm in lig_tr_atms:
        het_R = het_atm_R[hetatm]
        #
        het_CA_d = {}
        for resno in bsite_res_s:
            CA_R = bsite_CA_R[resno]
            d = dist_two_coords(het_R,CA_R)
            het_CA_d[resno] = d
        #
        sorted_ds = sorted(het_CA_d.items(), key=lambda item: item[1])
        closest = sorted_ds[0]
        closest_resno = closest[0]
        closest_d = closest[1]
        #
        
        cst_line = f'AtomPair {hetatm} {het_resno} CA {closest_resno} HARMONIC {closest_d} {CST_STDERR}'
        cst_s.append(cst_line)
    return cst_s
        
                
                    
    
