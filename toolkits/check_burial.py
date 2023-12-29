#!/usr/bin/env python
"""
from Gyu Rie
Linna Added the function to accept silent
"""
import os
import sys
import copy
import math
import pyrosetta as py
#GRL, 2021
from optparse import OptionParser
import pyrosetta
import pyrosetta.distributed.io as io
import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
import pyrosetta.distributed.tasks.score as score
import numpy as np
from pyrosetta import *
from pyrosetta.rosetta import *
# import pyrosetta
# import pyrosetta.distributed.io as io
# import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
sys.path.append( '/home/linnaan/software/silent_tools' )
import silent_tools

parser = OptionParser(usage="usage: %prog [options] FILE", version="0.1")
parser.add_option("--pdb", type="string", dest="pdb", metavar="STR", help="Input dataframe")
parser.add_option("--silent", type="string", dest="silent", metavar="STR", help="Input silent file")
parser.add_option("--tags", type="string", dest="tags", help="Input tags, connect with comma, technically this arguments is not needed, it is best to slice the selected silents to one file which make extraction faster")
parser.add_option("--lig_atom_names", type="string", dest="lig_atom_names", metavar="STR", help="comma seperated vector of atom names for which to measure buried")
parser.add_option("--params_file", type="string", dest="params_file", metavar="STR", help="Params")
parser.add_option("--out_sc_file", type="string", dest="out_sc_file", metavar="STR", help="outdir")
parser.add_option("--dump_test", type="string", dest="dump_test", metavar="STR", help="outdir")

(opts, args) = parser.parse_args()


def read_lig_atm_sasa(pose,atmName_to_read = [], probe_radius=1.4):
    atm_depth_s = py.rosetta.core.scoring.atomic_depth.atomic_depth(pose,probe_radius)
    #
    lig_resNo = pose.total_residue()
    lig_res = pose.residue(lig_resNo)
    res_natm = pose.pdb_info().natoms(lig_resNo)
    #this returns all atom values in list for the residue
    #atm_depth_s[lig_resNo]
    #
    tmp_atm_names = []
    for i_atm_loc in range(1, res_natm+1):
        atm_name = lig_res.atom_name(i_atm_loc)
        tmp_atm_names.append(atm_name)
    #
    atmNames = []
    if len(atmName_to_read) > 0:
        for atmName in tmp_atm_names:
            if atmName.strip() in atmName_to_read:
                atmNames.append(atmName)
    else:
        atmNames = copy.deepcopy(tmp_atm_names)
    #
    atom_depths = []
    for atm_name in atmNames:
        atm_id = py.rosetta.core.id.AtomID(lig_res.atom_index('%s'%atm_name), lig_resNo)
        atom_depths.append(atm_depth_s(atm_id))
    return np.sum(atom_depths)

def report_sasa(cont,header,outdir):
    fout = open('%s/%s_atmdepth.txt'%(outdir,header),'w')
    fout.write('%s\n'%('\n'.join(cont)))
    fout.close()
    return

def lig_pack_xml():
    xml = """
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
        <ScoreFunction name="sfxn" weights="ref2015">
          <Reweight scoretype="fa_intra_rep" weight="0.55"/>
        </ScoreFunction>
        <ScoreFunction name="sfxn_soft" weights="ref2015">
          <Reweight scoretype="fa_intra_rep" weight="0.55"/>
          <Reweight scoretype="fa_rep" weight="0.2"/>
        </ScoreFunction>
      </SCOREFXNS>
  
      <RESIDUE_SELECTORS>
    	<Chain name="chainA" chains="A"/>
        <Chain name="chainB" chains="B"/>
        <Neighborhood name="lig_neigh" distance="15.0">
          <Chain chains="B"/>
        </Neighborhood>  
        <Or name="interface_and_ligand" selectors="lig_neigh,chainB"/>
        <Not name="dont_pack_group" selector="interface_and_ligand"/>
      </RESIDUE_SELECTORS>
  
      <TASKOPERATIONS>
          <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" />
          <OperateOnResidueSubset name="fix_non_interface" selector="dont_pack_group"> 
              <PreventRepackingRLT/>
          </OperateOnResidueSubset>
          <OperateOnResidueSubset name="repack_interface" selector="interface_and_ligand">
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>
      </TASKOPERATIONS>
  
      <MOVERS>
        <PackRotamersMover name="pack_for_orient" scorefxn="sfxn"  task_operations="fix_non_interface,repack_interface"/>  ex1_ex2aro
      </MOVERS>
  
      <FILTERS>
     </FILTERS>
 
     <PROTOCOLS>
       <Add mover="pack_for_orient"/>
    </PROTOCOLS>
 
    </ROSETTASCRIPTS>
    """
    return xml

def main():

    params_file = opts.params_file
    py.init(f'-extra_res_fa {params_file} -keep_input_protonation_state ') #-mute all

    if opts.pdb:
        pdbname = opts.pdb
        pose0 = py.pose_from_pdb(pdbname)
        xml = lig_pack_xml()
        task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
        task_relax.setup() # syntax check
        atmNames = str(opts.lig_atom_names).split(',')
        
        pose = pose0.clone()
        packed_pose = task_relax(pose)
        total_atom_depth = read_lig_atm_sasa(pose, atmName_to_read=atmNames, probe_radius=3.0)
        print(f'Total buried depth {total_atom_depth}')
        
        with open(f'{opts.out_sc_file}', 'a+') as f_out:
            f_out.write('description total_atom_depth\n')
            f_out.write(f'{pdbname.split("/")[-1][:-4]} {total_atom_depth}\n')
        
        # header = pdbname.split('/')[-1].split('.pdb')[0]
        # outdir = '/net/scratch/gyurie/RO/roc_pack_orient/terminal_atmDepth_s'
        # report_sasa(cont,header,outdir)
        # return
        
        if opts.dump_test is not None:
            pose.dump_pdb('test.pdb')

    elif opts.silent:
        silent = opts.silent
        print(f'check {silent}')
        
        silent_index = silent_tools.get_silent_index(silent)
        sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
        sfd_in.read_file(silent)

        if opts.tags:
            tags =(opts.tags).split(',')
        else: 
            tags = list(silent_index['index'].keys())
        # print(f'TAGS {tags}')
        print(f'check {len(tags)} docks')

        with open(f'{opts.out_sc_file}', 'a+') as f_out:
                f_out.write('description total_atom_depth\n')

        for tag in tags:
            try:
                print(f' check {tag}')
                silent_structure = silent_tools.get_silent_structure( silent, silent_index, tag )
                pose0 = Pose()
                sfd_in.get_structure( tag ).fill_pose( pose0 )
                xml = lig_pack_xml()
                task_relax = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
                task_relax.setup() # syntax check
                atmNames = str(opts.lig_atom_names).split(',')
                
                pose = pose0.clone()
                packed_pose = task_relax(pose)
                total_atom_depth = read_lig_atm_sasa(pose, atmName_to_read=atmNames, probe_radius=3.0)
                print(f'Total buried depth {total_atom_depth}')
                
                with open(f'{opts.out_sc_file}', 'a+') as f_out:
                    f_out.write(f'{tag} {total_atom_depth}\n')
            except:
                print(f'Cannot measure for {tag}')

if __name__ == '__main__':
    main()
