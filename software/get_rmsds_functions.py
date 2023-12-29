import os, sys, glob

import subprocess, tempfile, os, re


def get_RMSD(tmout):
    #lines = tmout.splitlines()
    lines = [x.decode('utf-8') for x in tmout]
    rmsd_line = lines[16]
    rmsd = rmsd_line.split()[4].split(',')[0]
    return float(rmsd)

def get_tmscore(tmout):
    lines = tmout.splitlines()
    tmscore1_line = lines[17]
    tmscore2_line = lines[18]
    tmscore1 = float(tmscore1_line.split()[1])
    tmscore2 = float(tmscore2_line.split()[1])
    tmscore = ( tmscore1 + tmscore2 )/2
    return tmscore

def get_tmscore_ch1(tmout):
    lines = tmout.splitlines()
    tmscore1_line = lines[17]
    tmscore1 = float(tmscore1_line.split()[1])
    return tmscore1

def get_aligned_pos(tmout,print_result=True):
    lines = tmout.splitlines()
    top_PDB_seq_algn = lines[-4][:-1]
    charact = lines[-3][:-1]
    bottom_PDB_seq_algn = lines[-2][:-1]
    #print(top_PDB_seq_algn)
    #print(charact)
    #print(bottom_PDB_seq_algn)
    aligment_tuple_list = [ i for i in zip(top_PDB_seq_algn,charact,bottom_PDB_seq_algn)]
    nat_counter = 1
    des_counter = 1
    pos_align_tuple_list = []
    for tup in aligment_tuple_list:
        this_new_tup = ['-',tup[1],'-']
        if tup[0] != '-':
            this_new_tup[0] = nat_counter
            nat_counter += 1
        if tup[2] != '-':
            this_new_tup[2] = des_counter
            des_counter += 1
        pos_align_tuple_list.append(tuple(this_new_tup))
    if print_result:
        nat_counter = 1
        des_counter = 1
        pos_align_tuple_list_str = []
        for tup in aligment_tuple_list:
            this_new_tup_str = ['| - ',tup[1],'| - ']
            if tup[0] != '-':
               this_new_tup_str[0] = "|%03d"%nat_counter
               nat_counter += 1
            if tup[2] != '-':
               this_new_tup_str[2] = "|%03d"%des_counter
               des_counter += 1
            pos_align_tuple_list_str.append(tuple(this_new_tup_str))
        #print( ''.join([ i[0] for i in pos_align_tuple_list_str ] )+'|' )
        #print( ''.join([ '| '+i[1]+' ' for i in pos_align_tuple_list_str ] )+'|' )
        #print( ''.join([ i[2] for i in pos_align_tuple_list_str ] )+'|' )
    return pos_align_tuple_list

def TMalign(str1,str2,**kwargs):
    to_exec = "/home/drhicks1/bin/TMalign"

    args = [to_exec, str1, str2]
    
    process = subprocess.Popen(args, stdout=subprocess.PIPE)
    TM_out = process.stdout.readlines()

    return TM_out


# pdbs_file = sys.argv[1]

# with open(pdbs_file, "r") as f:
#     pdbs = f.readlines()
#     pdbs = [x.strip() for x in pdbs]

# for pdb in pdbs:
#     try:
#         base = os.path.splitext(os.path.basename(pdb))[0]

#         str1 = pdb
#         str2 = glob.glob(f"output/{base}*pdb")[0]

#         out = TMalign(str1,str2)
#         print(f"{base} rmsd: {get_RMSD(out)}")
#     except: print("")

