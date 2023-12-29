import os
import gzip
import glob
import sys
from numpy import *
from numpy import linalg
import subprocess

"""
usage: 
    close_hb_list,close_2aro_list,sat_lig_atm = check_interactions_to_lig.get_hb2lig(og_model,'FOL',LIG_heavy_atom,LIG_aro_atom)
"""
def get_lig_num(in_file,LIG):
    Lines=[]
    if in_file.endswith('.pdb.gz'):
        fp = gzip.open(in_file,'rt',encoding='utf-8')
        Lines = fp.readlines()
        fp.close()
    elif in_file.endswith('.pdb'):
        fp = open(in_file,'r')
        Lines = fp.readlines()
        fp.close()

    for line in Lines:
        if (LIG in line) and line.startswith('HETATM'):
            resNUM = int(line[22:26])
            break
    return resNUM

def get_hb2lig(pdb,lig_name,LIG_heavy_atom,LIG_aro_atom):
    """
    pyrosetta cannot get hbond to ligand correct. fix is too complicated, thus use distance to measure hb
    - check hb to LIG_heavy_atom, only count sc to ligand hb
    - check aromatic residues which are close to aromatic_atm_list, they are potential pi-pi interaction residues
    """
    input_pdb_1 = []
    if pdb.endswith('.gz'):
        with gzip.open(pdb,'rt') as f:
            for i in f:
                if i.startswith('ATOM') or i.startswith('HETATM'):
                    input_pdb_1.append(i.rstrip())
    elif pdb.endswith('.pdb'):
        with open(pdb,'r') as f:
            for i in f:
                if i.startswith('ATOM') or i.startswith('HETATM'):
                    input_pdb_1.append(i.rstrip())
#    print(input_pdb_1)
    vec1,vec2 = [],[]

    atom_type1,atom_id1,atom_name1,alt_loc1,res_name1,chain_name1,residue_id1,icode1,occupancy1,temp_factor1,segment1,element1,charge1= [],[],[],[],[],[],\
    [],[],[],[],[],[],[]
    
    aa_polar_atom_names = ("OG", "OG1", "OE1", "OE2","NE1", "NE2", "OD1", "OD2", "ND2", "OH", "ND1", "NH1", "NH2", "NE" ) #remove "O2", "N2",
    aa_aro_atom = ("CG","CD1","CD2","CE1","CE2","CZ","CE3","CZ2","CZ3","CH2")
    aro_aa = ("PHE","TYR","TRP","HIS")
    polar_atom_names = aa_polar_atom_names + LIG_heavy_atom
    aro_atom_names = LIG_aro_atom + aa_aro_atom

    collect_atoms = polar_atom_names + aro_atom_names

    # assign coordinate lines into different variables
    for line in input_pdb_1:
     #   print(line)
        if (line[12:17].strip() in collect_atoms):
            atom_type1.append(line[0:6].strip())
            atom_id1.append(line[6:11].strip())
            atom_name1.append(line[12:17].strip())
            alt_loc1.append(line[16:17].strip())
            res_name1.append(line[17:20].strip())
            chain_name1.append(line[21:22].strip())
            residue_id1.append(int(line[22:26].strip()))
            icode1.append(line[26:27].strip())
            occupancy1.append(float(line[55:60].strip()))
            temp_factor1.append(float(line[60:66].strip()))
            segment1.append(line[72:76].strip())
            element1.append(line[76:78].strip())
            charge1.append(line[78:80].strip())
            x1=float(line[30:38].strip())
            y1=float(line[38:46].strip())
            z1=float(line[46:54].strip())
            vec1.append(array([x1,y1,z1]))

    close_hb_list,sat_lig_atm,close_2aro_list,buttress_list = set(),set(),set(),set()

    #get polar res
    for i in range(len(vec1)):
        if (res_name1[i] == lig_name) and (atom_name1[i] in LIG_heavy_atom):
            for j in range(len(vec1)):
                if residue_id1[i] != residue_id1[j] and ((atom_name1[j] in polar_atom_names)):
                    dist = linalg.norm(vec1[i]-vec1[j])
                    if 0.1 < dist < 3.5 :
                        close_hb_list.add(int(residue_id1[j]))
                        sat_lig_atm.add(atom_name1[i])
    close_hb_list = sorted(close_hb_list)

    #check of around benzo ring of FOL, at least there are 5 close res
    for i in range(len(vec1)):
        if (res_name1[i] == lig_name) and ((atom_name1[i] in LIG_aro_atom)):
            for j in range(len(vec1)):
                if residue_id1[i] != residue_id1[j] and (res_name1[j] in aro_aa):
                    dist = linalg.norm(vec1[i]-vec1[j])
                    if 0.1 < dist < 4.1 : #pi-pi interaction limit 4.96, ref PMC5818208
                        close_2aro_list.add(int(residue_id1[j]))
    close_2aro_list = sorted(close_2aro_list)

    #get polar buttres residues
    for i in range(len(vec1)):
        if (residue_id1[i] in close_hb_list) and (atom_name1[i] in aa_polar_atom_names):
            for j in range(len(vec1)):
                if (residue_id1[i] != residue_id1[j]) and ((atom_name1[j] in aa_polar_atom_names)) and (res_name1[j] != lig_name):
                    dist = linalg.norm(vec1[i]-vec1[j])
                    if 0.1 < dist < 3.5 :
                        buttress_list.add(int(residue_id1[j]))
    for i in range(len(vec1)):
        if (residue_id1[i] in close_2aro_list) and (atom_name1[i] in aa_polar_atom_names):
            for j in range(len(vec1)):
                if (residue_id1[i] != residue_id1[j]) and ((atom_name1[j] in aa_polar_atom_names)) and (res_name1[j] != lig_name):
                    dist = linalg.norm(vec1[i]-vec1[j])
                    if 0.1 < dist < 3.5 :
                        buttress_list.add(int(residue_id1[j]))

    buttress_list = sorted(buttress_list)

    return close_hb_list,close_2aro_list,sat_lig_atm,buttress_list

def write_cst(LIG_heavy_atom,LIG_aro_atom,LIG_tail,input_file_1,LIG):
    '''
    - input_file_1 - the generated
    - generate cst to heavy atoms and aromatic atoms
    '''
#     input_pdb_1=open(input_file_1,'r').readlines()
    input_pdb_1 = []
    if input_file_1.endswith('.pdb.gz'):
        with gzip.open(input_file_1,'rt') as f:
            for line in f:
                input_pdb_1.append(line)
    elif input_file_1.endswith('.pdb'):
        with open(input_file_1,'r') as f:
            for line in f:
                input_pdb_1.append(line)

    vec1,vec2 = [],[]

    atom_type1,atom_id1,atom_name1,alt_loc1,res_name1,chain_name1,residue_id1,icode1,occupancy1,temp_factor1,segment1,element1,charge1= [],[],[],[],[],[],\
    [],[],[],[],[],[],[]

    cst_list = []

    polar_atom_names2 = ("OG", "OG1", "OE1", "OE2","NE1", "NE2", "OD1", "OD2", "ND2", "OH", "ND1", "NH1", "NH2", "NE" ) #remove "O2", "N2",
    aa_aro_atom = ("CG","CD1","CD2","CE1","CE2","CZ","CE3","CZ2","CZ3","CH2")
    polar_atom_names = polar_atom_names2 + LIG_heavy_atom
    aro_atom_names = LIG_aro_atom + aa_aro_atom
    bb_polar_atoms = ("O", "N")

    collect_atoms = bb_polar_atoms + polar_atom_names + aro_atom_names + ('CB',) + LIG_tail

    # assign coordinate lines into different variables
    for line in input_pdb_1:
        if (line[12:17].strip() in collect_atoms):
            atom_type1.append(line[0:6].strip())
            atom_id1.append(line[6:11].strip())
            atom_name1.append(line[12:17].strip())
            alt_loc1.append(line[16:17].strip())
            res_name1.append(line[17:20].strip())
            chain_name1.append(line[21:22].strip())
            residue_id1.append(int(line[22:26].strip()))
            icode1.append(line[26:27].strip())
#             occupancy1.append(float(line[55:60].strip()))
#             temp_factor1.append(float(line[60:66].strip()))
#             segment1.append(line[72:76].strip())
            element1.append(line[76:78].strip())
            charge1.append(line[78:80].strip())

            x1=float(line[30:38].strip())
            y1=float(line[38:46].strip())
            z1=float(line[46:54].strip())

            vec1.append(array([x1,y1,z1]))

    close_hb_list,close_res_list_remove_redundant,close_2aro_list,close_2aro_list_remove_redundant,divalent_list = [],[],[],[],[]

    #get polar res
    for i in range(len(vec1)):
        if res_name1[i] == LIG:
            for j in range(len(vec1)):
                if residue_id1[i] != residue_id1[j] and ((atom_name1[j] in polar_atom_names)):
                    dist = linalg.norm(vec1[i]-vec1[j])
                    if 0.1 < dist < 4 :
                        cst_list.append("AtomPair {} {} {} {} HARMONIC {:3.2f} 0.2".\
                                        format(atom_name1[i],residue_id1[i],atom_name1[j],residue_id1[j],dist))
                        close_hb_list.append(int(residue_id1[j]))

    for i in close_hb_list:
        if i not in close_res_list_remove_redundant:
            close_res_list_remove_redundant.append(i)
    close_res_list_remove_redundant.sort()

    #check of around benzo ring of FOL, at least there are 5 close res
    for i in range(len(vec1)):
        if res_name1[i] == LIG and ((atom_name1[i] in aro_atom_names)):
            for j in range(len(vec1)):
                if residue_id1[i] != residue_id1[j]:
                    dist = linalg.norm(vec1[i]-vec1[j])
                    if 0.1 < dist < 4.5 :
                        cst_list.append("AtomPair {} {} {} {} HARMONIC {:3.2f} 0.2".\
                                        format(atom_name1[i],residue_id1[i],atom_name1[j],residue_id1[j],dist))
                        close_2aro_list.append(int(residue_id1[j]))

    return cst_list

def get_contact_residues(pdb,lig_name,LIG_atom,distance_cutoff):
    """
    give the contact residues within cutoff to the chosen ligand atoms, use this as a way to check burial
    - check hb to LIG_heavy_atom, only count sc to ligand hb
    - check aromatic residues which are close to aromatic_atm_list, they are potential pi-pi interaction residues
    """
    input_pdb_1 = []
    if pdb.endswith('.gz'):
        with gzip.open(pdb,'rt') as f:
            for line in f:
                input_pdb_1.append(line)
    elif pdb.endswith('.pdb'):
        with open(pdb,'r') as f:
            for line in f:
                input_pdb_1.append(line)

    vec1,vec2 = [],[]

    atom_type1,atom_id1,atom_name1,alt_loc1,res_name1,chain_name1,residue_id1,icode1,occupancy1,temp_factor1,segment1,element1,charge1= [],[],[],[],[],[],\
    [],[],[],[],[],[],[]
    
    polar_atom_names2 = ("OH","OG", "OG1", "OE1", "OE2","NE1", "NE2","NZ","OD1", "OD2","ND2", "OH", "ND1", "NH1", "NH2", "NE" ) #remove "O2", "N2",
    aa_aro_atom = ("CG","CD1","CD2","CE1","CE2","CZ","CE3","CZ2","CZ3","CH2")
    aa_norm_atom = ("CG","CE","SD","CG1","CG2",) #missing W H
    polar_atom_names = polar_atom_names2 
    aro_atom_names = aa_aro_atom
    bb_polar_atoms = ("O","N")

    collect_atoms = bb_polar_atoms + polar_atom_names + aro_atom_names + LIG_atom + aa_norm_atom + ("CA","C","CB")

    # assign coordinate lines into different variables
    for line in input_pdb_1:
        if (line[12:17].strip() in collect_atoms):
            atom_type1.append(line[0:6].strip())
            atom_id1.append(line[6:11].strip())
            atom_name1.append(line[12:17].strip())
            alt_loc1.append(line[16:17].strip())
            res_name1.append(line[17:20].strip())
            chain_name1.append(line[21:22].strip())
            residue_id1.append(int(line[22:26].strip()))
            icode1.append(line[26:27].strip())
            occupancy1.append(float(line[55:60].strip()))
            temp_factor1.append(float(line[60:66].strip()))
            segment1.append(line[72:76].strip())
            element1.append(line[76:78].strip())
            charge1.append(line[78:80].strip())
            x1=float(line[30:38].strip())
            y1=float(line[38:46].strip())
            z1=float(line[46:54].strip())
            vec1.append(array([x1,y1,z1]))

    close_list = set()

    #get residues by contact
    for i in range(len(vec1)):
        if (res_name1[i] == lig_name) and (atom_name1[i] in LIG_atom):
            for j in range(len(vec1)):
                if (residue_id1[i] != residue_id1[j]) and ((atom_name1[j] in collect_atoms)):
                    dist = linalg.norm(vec1[i]-vec1[j])
                    if 0.1 < dist < distance_cutoff :
                        close_list.add(int(residue_id1[j]))
    close_list = sorted(close_list)

    return close_list
