#!/usr/bin/env python

"""
This script is used to perform full design to small molecule docks.
Prcoedure:
0. add bb cst and cst ligand to closest residue bb
1. first fast design, freeze rifres, ligand, pack core to non-charge, pack interface to all aa
2. second fast design, free rifres, ligand, pack core to non-charge, pack interface to all aa
3. relax and score
Version: 2022.04.13

edit
Prcoedure:
0. remove ligand cst during relaxing
1. no longer do add 2 stage FastDesign (without design), to save time, previously these steps were kept to make scores easily comparable to Rosetta design
1. add rotamer boltzman
Version: 2022.04.13

Author: Linna An
"""
import os, sys
import math
import numpy as np
from collections import defaultdict
import time
import argparse
import itertools
import subprocess
import time
import pandas as pd
import glob
from decimal import Decimal
from collections import OrderedDict

# sys.path.insert( 0, '/home/nrbennet/rosetta_builds/master_branch/pyrosetta_builds/py39_builds/build1' )
# sys.path.insert(0, '/home/nrbennet/protocols/dl/dl_design/justas_seq_op/single_chain/')
sys.path.append( '/home/linnaan/software/silent_tools' )
import silent_tools
sys.path.append( '/home/linnaan/lib/' )
from libCommonJupyterFunc import get_total_scores

from pyrosetta import *
from pyrosetta.rosetta import *
# import pyrosetta
# import pyrosetta.distributed.io as io
# import pyrosetta.distributed.packed_pose as packed_pose
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
# import pyrosetta.distributed.tasks.score as score

#add for use apptainer
# export LD_LIBRARY_PATH=/software/rosetta
os.environ['LD_LIBRARY_PATH']='/software/rosetta'
#####################################################################################
# input
#####################################################################################   
argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argparser.add_argument( "--out_name", type=str, default="out", help='The name of your out silent file' )
argparser.add_argument( "--silent", type=str, default="", help='The name of a silent file to run this metric on. pdbs are not accepted at this point in time' )
argparser.add_argument( "--pdb", type=str, default="", help='The name of a pdb file to run this metric on.' )
argparser.add_argument( "--params", type=str, default="params", help='params file for the ligand' )
argparser.add_argument( "--ligand_res_number", type=str, default="", help='ligand res number' )
#argparser.add_argument( "--key_contacts", type=str, default="", help='contact residues to freeze in first round of design, connected by comma' )
argparser.add_argument( "--heavy_atms", type=str, default="", help='ligand heavy atms, connected by comma' )
argparser.add_argument( "--tags", type=str, default="", help='the tags of the pdbs to design, connect by comma' )
argparser.add_argument( "--suffix", type=str, default="", help='suffix to add, need to include _ sign' )
argparser.add_argument( "--save_to_pdb", type=bool, default="False", help='save to pdb or silent' )
argparser.add_argument( "--save_to_silent", type=bool, default="False", help='save to pdb or silent' )
argparser.add_argument( "--save_score_only", type=bool, default="True", help='do not save pdb, only sc' )
argparser.add_argument( "--use_genpot", type=bool, default="False", help='use genpot? default use beta_nov16' )
argparser.add_argument( "--cache", type=str, default="/net/scratch/linnaan/cache/", help='cache directory for psipred SSpred etc' )
args = argparser.parse_args()    

silent = args.silent###
if args.silent:
    tags = args.tags.split(',')
in_pdb = args.pdb
if in_pdb:
    tags = [os.path.basename(in_pdb).replace('.pdb','')]
ligand_res_number = args.ligand_res_number
heavy_atms = args.heavy_atms.split(',')
param_f = args.params
SUFFIX = args.suffix
cache_dir = args.cache

#####################################################################################
# Initialize pose
#####################################################################################
print(f'Input ligand_res {ligand_res_number}')

if args.use_genpot:
    pyrosetta.init(f"-beta -holes:dalphaball /software/rosetta/DAlphaBall.gcc -use_terminal_residues true -mute basic.io.database core.scoring -dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8 -dunbrack_prob_buried_semi 0.8 -dunbrack_prob_nonburied_semi 0.8 -no_nstruct_label true -nstruct 1 -precompute_ig -out:path:scratch {cache_dir} -out:file:write_pdb_parametric_info True -out:file:scorefile score.sc -extra_res {param_f} -run:preserve_header ") #-indexed_structure_store:fragment_store /databases/vall/ss_grouped_vall_helix_shortLoop.h5 
else:
    pyrosetta.init(f"-beta_nov16 -corrections:beta_nov16 -holes:dalphaball /software/rosetta/DAlphaBall.gcc -use_terminal_residues true -mute basic.io.database core.scoring -dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8 -dunbrack_prob_buried_semi 0.8 -dunbrack_prob_nonburied_semi 0.8 -no_nstruct_label true -nstruct 1 -precompute_ig -out:path:scratch {cache_dir} -out:file:write_pdb_parametric_info True -out:file:scorefile score.sc -extra_res {param_f} -run:preserve_header ") #-indexed_structure_store:fragment_store /databases/vall/ss_grouped_vall_helix_shortLoop.h5 

print(f'inited!')

def design(pose, tag, sfd_out):

    t0 = time.time()

    #####################################################################################
    # functions
    #####################################################################################
    # def to_pdb_file(final_pose, filename):
    #     print(f'start to save PDB')
    #     print(f'pose string:\n' + pyrosetta.distributed.io.to_pdbstring(final_pose))
    #     with open(filename, "w+") as opdb:
    #         opdb.write(pyrosetta.distributed.io.to_pdbstring(final_pose))
    #     print('save to PDB success!')

    def add2silent( pose, tag, sfd_out ):
        struct = sfd_out.create_SilentStructOP()
        struct.fill_struct( pose, tag )
        sfd_out.add_structure( struct )
        sfd_out.write_silent_struct( struct, silent_out )
        print('save to silent')

    def get_all_close_res(pose, ligand_res_number):
        """
        get cst for ligand
        atm_list - one ligand atms, use for estimate distance
        usage:
        cst_list = get_all_atom_close_csts(pose, bb_only=False, bb_sd=0.5, sc_sd=10.0)

            for cst in cst_list:
                pose.add_constraint(cst)
        """
        # bbs = ["N", "O", "C", "CA", "CB"]
        ligand = int(ligand_res_number)
        close_res = []
        close_dist_cutoff = 5
        for resi in range(1, pose.size()):
            for at_i in range(1, pose.residue(resi).natoms() + 1):
                for at_j in range(1, pose.residue(ligand).natoms() + 1):
                    if pose.residue(resi).xyz("CA").distance_squared(pose.residue(ligand).xyz(pose.residue(ligand).atom_name(at_j).strip())) >= 100:
                        continue
                    id_i = pyrosetta.rosetta.core.id.AtomID(at_i, resi)
                    id_j = pyrosetta.rosetta.core.id.AtomID(at_j, ligand)

                    i_j_dist = pose.residue(resi).xyz(at_i).distance(pose.residue(ligand).xyz(at_j))

                    if (i_j_dist < close_dist_cutoff) and (resi not in close_res):
                        close_res.append(resi)
        if ligand in close_res:
            close_res.remove(ligand)

        print(f'{len(close_res)} close_res {close_res} generated')

        return ','.join(str(i) for i in set(close_res))

    def get_all_atom_close_csts(pose, ligand_res_number, bb_only=False, sd=1.0, no_ligand_cst=False):
        """
        get cst for ligand
        atm_list - one ligand atms, use for estimate distance
        usage:
        cst_list = get_all_atom_close_csts(pose, bb_only=False, bb_sd=0.5, sc_sd=10.0)

            for cst in cst_list:
                pose.add_constraint(cst)
        """
        # bbs = ["N", "O", "C", "CA", "CB"]
        cst_list = []
        ligand = int(ligand_res_number)
        if no_ligand_cst == False:
            for resi in range(1, pose.size()):
                for at_i in range(1, pose.residue(resi).natoms() + 1):
                    if pose.residue(resi).atom_name(at_i).strip() != "CA": #only do CA, could do CB instead but then need logic for GLY
                        continue    
                    best_dist = 11
                    cst = ""
                    for at_j in range(1, pose.residue(resj).natoms() + 1):
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
        print(f'after generating protein-ligand cst {len(cst_list)}')
        for resi in range(1, pose.size()):
            for resj in range(1, pose.size()):
                for at_i in range(1, pose.residue(resi).natoms() + 1):
                    if pose.residue(resi).atom_name(at_i).strip() != "CA": #only do CA, could do CB instead but then need logic for GLY
                        continue    
                    best_dist = 11
                    cst = ""
                    for at_j in range(1, pose.residue(resj).natoms() + 1):
                        if pose.residue(resj).atom_name(at_j).strip() != "CA": #only do CA, could do CB instead but then need logic for GLY
                            continue 
                        elif pose.residue(resi).xyz("CA").distance_squared(pose.residue(resj).xyz("CA")) >= 100:
                            continue
                        id_i = pyrosetta.rosetta.core.id.AtomID(at_i, resi)
                        id_j = pyrosetta.rosetta.core.id.AtomID(at_j, resj)

                        i_j_dist = pose.residue(resi).xyz(at_i).distance(pose.residue(resj).xyz(at_j))

                        if i_j_dist < best_dist:
                            best_dist = i_j_dist
                            func = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(best_dist, sd)
                            cst = pyrosetta.rosetta.core.scoring.constraints.AtomPairConstraint(id_i, id_j, func)
                    # if cst != "":
                            cst_list.append(cst)

        print(f'Add protein bb cst, in total {len(cst_list)} cst generated')

        return cst_list


    # def calc_rboltz(pose,close_res):
    #     '''
    #     Takes in a pose and the existing DataFrame of scores for a design and returns the DataFrame with
    #     three new columns: largest RotamerBoltzmann of ARG/LYS/GLU/GLN residues; average of the top two
    #     RotamerBoltzmann scores (includes every amino acid type); and median RotamerBoltzmann (includes every amino acid type)
    #     '''
    #     notable_aas = ['ARG','LYS','HIS','GLU','ASP','GLN','SER','THR','TYR','TRP','PHE']
    #     hbond_residues, bidentates = count_hbonds_protein_dna(pose)

    #     cols = ['residue_num','residue_name','rboltz']
    #     design_df = pd.DataFrame(columns=cols)

    #     for j in hbond_residues:
    #         residue_info = [j, pose.residue(j).name()[:3]]

    #         hbond_position_selector = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    #         hbond_position_selector.set_index(j)

    #         # Set up task operations for RotamerBoltzmann...with standard settings
    #         tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    #         tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    #         tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    #         tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    #         # Allow extra rotamers
    #         extra_rots = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
    #         extra_rots.ex1(1)
    #         extra_rots.ex2(1)
    #         tf.push_back(extra_rots)

    #         # Prevent repacking on everything but the hbond position
    #         prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
    #         not_pack = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(hbond_position_selector)
    #         prevent_subset_repacking = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking_rlt, not_pack)
    #         tf.push_back(prevent_subset_repacking)

    #         sfxn = pyrosetta.rosetta.core.scoring.get_score_function()

    #         rboltz = pyrosetta.rosetta.protocols.calc_taskop_filters.RotamerBoltzmannWeight()
    #         rboltz.scorefxn(sfxn)
    #         rboltz.task_factory(tf)
    #         rboltz.skip_ala_scan(1)
    #         rboltz.no_modified_ddG(1)
    #         rboltz_val = rboltz.compute(pose)
    #         residue_info.append(rboltz_val)
    #         design_df.loc[len(design_df)] = residue_info.copy()

    #     RKQE_subset = design_df[design_df['residue_name'].isin(notable_aas)]
    #     # if len(RKQE_subset) > 0:
    #     #     df['max_rboltz_RKQE'] = -1 * RKQE_subset['rboltz'].min()
    #     # else:
    #     #     df['max_rboltz_RKQE'] = 0

    #     # if len(design_df) > 0:
    #     #     df['avg_top_two_rboltz'] = -1 * np.average(design_df['rboltz'].nsmallest(2))
    #     #     df['median_rboltz'] = -1 * np.median(design_df['rboltz'])
    #     # else:
    #     #     df['avg_top_two_rboltz'] = 0
    #     #     df['median_rboltz'] = 0

    #     return design_df

    def generate_hb_filters(atm_list,scorefxn,ligand_res_number):
        filters,protocols = [],[]
        for atm in atm_list:
            filters.append(f'<SimpleHbondsToAtomFilter name="hb_to_{atm}" n_partners="1" hb_e_cutoff="-0.3" target_atom_name="{atm}" res_num="{ligand_res_number}" scorefxn="{scorefxn}" confidence="0"/>')
            protocols.append(f'<Add filter="hb_to_{atm}"/>')
        return '\n'.join(filters),'\n'.join(protocols)
    
    filters,protocols = generate_hb_filters(heavy_atms,'sfxn',ligand_res_number)
    print(f'generated filters')
    
    xml = f"""
    <ROSETTASCRIPTS>  
    #this protocol moves the ligand too much during minimization, transfer this to
        <SCOREFXNS>
            <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="res_type_constraint" weight="0.3"/>
              <Reweight scoretype="arg_cation_pi" weight="3"/>
          <Reweight scoretype="approximate_buried_unsat_penalty" weight="5"/>
              <Set approximate_buried_unsat_penalty_burial_atomic_depth="3.5"/>
              <Set approximate_buried_unsat_penalty_hbond_energy_threshold="-0.5"/>
              <Set approximate_buried_unsat_penalty_hbond_bonus_cross_chain="-1"/>
              <Reweight scoretype="atom_pair_constraint" weight="0.3"/>
              <Reweight scoretype="dihedral_constraint" weight="0.1"/>
              <Reweight scoretype="angle_constraint" weight="0.1"/>
              <Reweight scoretype="aa_composition" weight="1.0" />
            </ScoreFunction>
            <ScoreFunction name="sfxn" weights="beta"/>    
            <ScoreFunction name="sfxn_softish" weights="beta">
                <Reweight scoretype="fa_rep" weight="0.15" />
            </ScoreFunction>
        <ScoreFunction name="vdw_sol" weights="empty" >
          <Reweight scoretype="fa_atr" weight="1.0" />
          <Reweight scoretype="fa_rep" weight="0.55" />
          <Reweight scoretype="fa_sol" weight="1.0" />
        </ScoreFunction>
          </SCOREFXNS>
          
          <RESIDUE_SELECTORS>
            <Layer name="init_core_SCN" select_core="True" use_sidechain_neighbors="True" surface_cutoff="1.0" /> 
            <Layer name="init_boundary_SCN" select_boundary="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
            <Layer name="surface_SCN" select_surface="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
            <Layer name="coreRes" select_core="true" use_sidechain_neighbors="true" core_cutoff="2.1" surface_cutoff="1.0"/>
            <ResiduePDBInfoHasLabel name="hbnet_res" property="HBNet" />
            Not name="not_hbnet_res" selector="hbnet_res" /> 
            And name="surface_SCN_and_not_hbnet_res" selectors="surface_SCN,not_hbnet_res"/>
            <ResidueName name="select_AVLI" residue_names="ALA,VAL,LEU,ILE" />
            <Not name="not_AVLI" selector="select_AVLI" />
            <ResiduePDBInfoHasLabel name="all_rifres_res" property="RIFRES"/>
            <And name="rifres_res" selectors="all_rifres_res,not_AVLI" />
            <Chain name="chainA" chains="A"/>
            <Chain name="chainB" chains="B"/>
        <Index name="ligand" resnums="{ligand_res_number}"/>
        <Not name="not_ligand" selector="ligand"/>
        <CloseContact name="interface_by_contact" residue_selector="ligand" contact_threshold="8" /> /this will select the ligand as well
        <And name="interface" selectors="interface_by_contact,not_ligand"/>
            <Not name="not_interface" selector="interface"/>
            <Or name="interface_and_ligand" selectors="interface,ligand"/>
            <And name="not_interface_or_ligand" selectors="chainA,not_interface" />
            <ResidueName name="select_polar" residue_names="GLU,ASP,ARG,HIS,GLN,ASN,THR,SER,TYR,TRP" />
            <ResidueName name="select_PG" residue_names="PRO,GLY" />
            #layer design definition
            Layer name="surface" select_core="false" select_boundary="false" select_surface="true" use_sidechain_neighbors="true"/>
            Layer name="boundary" select_core="false" select_boundary="true" select_surface="false" use_sidechain_neighbors="true"/>
            <Layer name="core_by_SC" select_core="true" select_boundary="false" select_surface="false" use_sidechain_neighbors="true" core_cutoff="5.2"/>
            <Layer name="core_by_SASA" select_core="true" select_boundary="false" select_surface="false" use_sidechain_neighbors="false" ball_radius="2" core_cutoff="20"/>
        <Or name="core_by_SC_SASA" selectors="core_by_SC,core_by_SASA"/>
        <And name="core" selectors="core_by_SC_SASA,not_ligand"/>
            <Not name="not_core" selector="core"/>
            <And name="not_core_chA" selectors="not_core,chainA,not_interface_or_ligand"/>
            <SecondaryStructure name="sheet" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="E"/>
            <SecondaryStructure name="entire_loop" overlap="0" minH="3" minE="2" include_terminal_loops="true" use_dssp="true" ss="L"/>
            <SecondaryStructure name="entire_helix" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="H"/>
            <And name="helix_cap" selectors="entire_loop">
              <PrimarySequenceNeighborhood lower="1" upper="0" selector="entire_helix"/>
            </And>
            <And name="helix_start" selectors="entire_helix">
              <PrimarySequenceNeighborhood lower="0" upper="1" selector="helix_cap"/>
            </And>
            <And name="helix" selectors="entire_helix">
              <Not selector="helix_start"/>
            </And>
            <And name="loop" selectors="entire_loop">
              <Not selector="helix_cap"/>
            </And>  
          </RESIDUE_SELECTORS>

          <RESIDUE_LEVEL_TASK_OPERATIONS>
            <PreventRepackingRLT name="PreventRepacking" />
            <RestrictToRepackingRLT name="RestrictToRepacking" />
          </RESIDUE_LEVEL_TASK_OPERATIONS>
          
          <TASKOPERATIONS>
              <SetIGType name="precompute_ig" lin_mem_ig="false" lazy_ig="false" double_lazy_ig="false" precompute_ig="true"/> 
              SeqprofConsensus name="pssm_cutoff" filename="%%pssmFile%%" min_aa_probability="-1" convert_scores_to_probabilities="0" probability_larger_than_current="0" debug="1" ignore_pose_profile_length_mismatch="1"/>
              <RestrictAbsentCanonicalAAS name="noCys" keep_aas="ADEFGHIKLMNPQRSTVWY"/>
              <PruneBuriedUnsats name="prune_buried_unsats" allow_even_trades="false" atomic_depth_cutoff="3.5" minimum_hbond_energy="-0.5" />
              <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
              <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" />
              <IncludeCurrent name="ic"/>

              <DesignRestrictions name="design_task">
    <!--             <Action selector_logic="surface AND helix_start"  aas="DEHKPQR"/>
                <Action selector_logic="surface AND helix"    aas="EHKQR"/>
                <Action selector_logic="surface AND sheet"    aas="EHKNQRST"/>
                <Action selector_logic="surface AND loop"   aas="DEGHKNPQRST"/>
                <Action selector_logic="boundary AND helix_start" aas="ADEHIKLMNPQRSTVWY"/>
                <Action selector_logic="boundary AND helix"   aas="ADEHIKLMNQRSTVWY"/>
                <Action selector_logic="boundary AND sheet"   aas="DEFHIKLMNQRSTVWY"/>
                <Action selector_logic="boundary AND loop"    aas="ADEFGHIKLMNPQRSTVWY"/>
                <Action selector_logic="surface"  residue_level_operations="PreventRepacking" /> -->
                <Action selector_logic="not_core_chA"  residue_level_operations="PreventRepacking" />
                <!-- <Action selector_logic="core AND helix_start"   aas="AFILMPVWY"/>
                <Action selector_logic="core AND helix"     aas="AFILMVWY"/>
                <Action selector_logic="core AND sheet"     aas="FILMVWY"/> -->
                <Action selector_logic="core NOT rifres_res"      aas="AFILMVWYSTQN"/>
            <Action selector_logic="rifres_res"      aas="AFILMVWYSTQNDERKH"/>
            <Action selector_logic="interface NOT core"      aas="AFILMVWYSTQNDERKH"/>
                <!-- <Action selector_logic="helix_cap"      aas="DNST"/> -->
              </DesignRestrictions>

              <OperateOnResidueSubset name="restrict_to_packing_not_interface" selector="not_interface"><RestrictToRepackingRLT/></OperateOnResidueSubset>
              <OperateOnResidueSubset name="restrict_to_interface" selector="not_interface_or_ligand"><PreventRepackingRLT/></OperateOnResidueSubset>
              <!-- <OperateOnResidueSubset name="ld_surface_not_hbnets" selector="surface_SCN_and_not_hbnet_res"><PreventRepackingRLT/></OperateOnResidueSubset> -->
              <OperateOnResidueSubset name="ld_surface" selector="surface_SCN"><PreventRepackingRLT/></OperateOnResidueSubset>
              <OperateOnResidueSubset name="restrict_packing_rifres_res" selector="rifres_res"><RestrictToRepackingRLT/></OperateOnResidueSubset>
              <OperateOnResidueSubset name="restrict_packing_interface" selector="interface"><RestrictToRepackingRLT/></OperateOnResidueSubset>
              <!-- <OperateOnResidueSubset name="fix_hbnet_residues" selector="hbnet_res"><RestrictToRepackingRLT/></OperateOnResidueSubset> -->
              <OperateOnResidueSubset name="restrict_target2repacking" selector="ligand"><PreventRepackingRLT/></OperateOnResidueSubset> #change from RestrictToRepackingRLT to PreventRepackingRLT
                  
              <ProteinProteinInterfaceUpweighter name="upweight_interface" interface_weight="3" />
              <ProteinInterfaceDesign name="pack_long" design_chain1="0" design_chain2="0" jump="1" interface_distance_cutoff="15"/>
          </TASKOPERATIONS>
          
          <MOVERS>
        <AddResidueLabel name="label_core" residue_selector="core" label="core" />
        <AddResidueLabel name="label_interface" residue_selector="interface" label="interface" />
            <AddConstraintsToCurrentConformationMover name="add_bb_cst" use_distance_cst="False" cst_weight="1" bb_only="1" sc_tip_only="0" />
            <ClearConstraintsMover name="rm_bb_cst" />
            <TaskAwareMinMover name="min" scorefxn="sfxn" bb="0" chi="1" task_operations="pack_long" />
            FavorSequenceProfile name="FSP" scaling="none" weight="1" pssm="%%pssmFile%%" scorefxns="sfxn_design"/>
            
            <!-- <PackRotamersMover name="hard_pack" scorefxn="sfxn_design"  task_operations="ex1_ex2aro,ld_surface_not_hbnets,fix_hbnet_residues,ic,limitchi2,pssm_cutoff,noCys,restrict_packing_rifres_res,upweight_interface,restrict_to_interface"/> 
            <TaskAwareMinMover name="softish_min" scorefxn="sfxn_softish" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="restrict_to_packing_not_interface" />
            <TaskAwareMinMover name="hard_min" scorefxn="sfxn" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="restrict_to_packing_not_interface" />  -->
                    
            <FastRelax name="fastRelax" scorefxn="sfxn" task_operations="ex1_ex2aro,ic"/>
            <ClearConstraintsMover name="rm_csts" />
            
            <FastDesign name="fastDesign_stage1" scorefxn="sfxn_design" repeats="1" task_operations="precompute_ig,design_task,ex1_ex2aro,ic,limitchi2,noCys,restrict_packing_rifres_res,upweight_interface,restrict_to_packing_not_interface,restrict_target2repacking,restrict_packing_interface" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"/> /do not design for interace, but allow relax
            
            <FastDesign name="fastDesign_stage2" scorefxn="sfxn_design" repeats="3" task_operations="precompute_ig,design_task,ex1_ex2aro,ic,limitchi2,noCys,upweight_interface,restrict_to_packing_not_interface,restrict_target2repacking,restrict_packing_interface" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"/> /do not design for interace, but allow relax
            
            MutateResidue name="install_protonated_his" residue_selector="the_hip" new_res="000" preserve_atom_coords="false" mutate_self="false"/>
            MutateResidue name="remove_protonated_his" residue_selector="the_hip" new_res="HIS" preserve_atom_coords="false" mutate_self="false"/>
            DumpPdb name="dump_test" fname="test.pdb" scorefxn="sfxn"/>
            
            <SwitchChainOrder name="chain1onlypre" chain_order="1" />
            <ScoreMover name="scorepose" scorefxn="sfxn" verbose="false" />
            <ParsedProtocol name="chain1only">
                <Add mover="chain1onlypre" />
                <Add mover="scorepose" />
            </ParsedProtocol>
            
            <!-- <ParsedProtocol name="chain1only_no_his_protonation">
                <Add mover="chain1onlypre" />
                Add mover="remove_protonated_his"/>
                <Add mover="scorepose" />
            </ParsedProtocol> -->
            
            <!-- <ParsedProtocol name="short_repack_and_min">
             <Add mover="hard_pack" />
             <Add mover="softish_min" />
             <Add mover="hard_min" />
            </ParsedProtocol> -->
            
            <DeleteRegionMover name="delete_polar" residue_selector="select_polar" rechain="false" />
            <SavePoseMover name="save_pose" restore_pose="0" reference_name="after_design" />
            
          </MOVERS>
          
          <FILTERS>
            <Rmsd name="lig_rmsd_after_final_relax" reference_name="after_design" superimpose_on_all="0" superimpose="1" threshold="5" confidence="0" >
               <span begin_res_num="{ligand_res_number}" end_res_num="{ligand_res_number}"/>
            </Rmsd>

            <MoveBeforeFilter name="move_then_lig_rmsd" mover="fastRelax" filter="lig_rmsd_after_final_relax" confidence="0" />

            <ScoreType name="totalscore" scorefxn="sfxn" threshold="9999" confidence="1"/>
            <ResidueCount name="nres" confidence="1" />
            <CalculatorFilter name="score_per_res" confidence="1" equation="SCORE/NRES" threshold="999">
                <Var name="SCORE" filter_name="totalscore" />
                <Var name="NRES" filter_name="nres" />
            </CalculatorFilter>

            <Geometry name="geometry" omega="165" cart_bonded="20" start="1" end="9999" count_bad_residues="true" confidence="0"/>

            <Ddg name="ddg_norepack"  threshold="0" jump="1" repeats="1" repack="0" relax_mover="min" confidence="0" scorefxn="sfxn"/>  
            <Report name="ddg1" filter="ddg_norepack"/>
            <Report name="ddg2" filter="ddg_norepack"/>


            <ShapeComplementarity name="SC" min_sc="0" min_interface="0" verbose="0" quick="0" jump="1" confidence="0"/>
            HbondsToResidue name="hbonds2lig" scorefxn="sfxn" partners="0" energy_cutoff="-0.5" backbone="0" bb_bb="0" sidechain="1" residue="{2}"/>
            BuriedUnsatHbonds2 name="interf_uhb2" cutoff="200" scorefxn="sfxn" jump_number="1"/>
            <ContactMolecularSurface name="contact_molecular_surface" distance_weight="0.5" target_selector="chainA" binder_selector="chainB" confidence="0" use_rosetta_radii="1"/>
            <Holes name="hole" threshold="20.0" residue_selector="coreRes" exclude_bb_atoms="false" />

            {filters}

            <Sasa name="interface_buried_sasa" confidence="0" />      

            <InterfaceHydrophobicResidueContacts name="hydrophobic_residue_contacts" target_selector="chainB" binder_selector="chainA" scorefxn="sfxn" confidence="0"/>

            <SSPrediction name="pre_mismatch_probability" confidence="0" cmd="/software/psipred4/runpsipred_single" use_probability="1" mismatch_probability="1" use_svm="0" use_scratch_dir="1"/>
            <MoveBeforeFilter name="mismatch_probability" mover="chain1only" filter="pre_mismatch_probability" confidence="0" />

            <SSPrediction name="pre_sspred_overall" cmd="/software/psipred4/runpsipred_single" use_probability="0" use_svm="0" threshold="0.85" confidence="0" use_scratch_dir="1" />
            <MoveBeforeFilter name="sspred_overall" mover="chain1only" filter="pre_sspred_overall" confidence="0" />
            <!-- <MoveBeforeFilter name="clash_check" mover="short_repack_and_min" filter="ddg1" confidence="0"/> -->
            <Ddg name="ddg_hydrophobic_pre"  threshold="-10" jump="1" repeats="1" repack="0" confidence="0" scorefxn="vdw_sol" />
            <MoveBeforeFilter name="ddg_hydrophobic" mover="delete_polar" filter="ddg_hydrophobic_pre" confidence="0"/>

            <ResidueCount name="nMET" count_as_percentage="1" max_residue_count="300" min_residue_count="0" residue_types="MET" confidence="0" />
            <ResidueCount name="nALA" count_as_percentage="1" max_residue_count="300" min_residue_count="0" residue_types="ALA" confidence="0" />
            <ResidueCount name="nARG" count_as_percentage="1" max_residue_count="300" min_residue_count="0" residue_types="ARG" confidence="0" />
            <ResidueCount name="nHIS" count_as_percentage="1" max_residue_count="300" min_residue_count="0" residue_types="HIS" confidence="0" />
            ResidueCount name="ala_loop_count" max_residue_count="300" residue_types="ALA" count_as_percentage="1" residue_selector="loop" confidence="0"/>
            <ResidueCount name="ala_core_count" max_residue_count="300" residue_types="ALA" count_as_percentage="1" residue_selector="core" confidence="0"/>
            ResidueCount name="ala_bdry_count" max_residue_count="300" residue_types="ALA" count_as_percentage="1" residue_selector="boundary" confidence="0"/>
            <ResidueCount name="res_count_all" max_residue_count="9999" confidence="0"/>
            <ScoreType name="hb_lr_bb" scorefxn="sfxn" score_type="hbond_lr_bb" confidence="0" threshold="0"/>
            <ResidueCount name="nres_all"/>
            <CalculatorFilter name="hb_lr_bb_per_res" equation="FAA/RES" threshold="0" confidence="0">
                    <Var name="FAA" filter="hb_lr_bb" />
                    <Var name="RES" filter="nres_all"/>
            </CalculatorFilter>
            <ScoreType name="hb_sr_bb" scorefxn="sfxn" score_type="hbond_sr_bb" confidence="0" threshold="0"/>
            <CalculatorFilter name="hb_sr_bb_per_res" equation="FAA/RES" threshold="0" confidence="0">
                    <Var name="FAA" filter="hb_sr_bb" />
                    <Var name="RES" filter="nres_all"/>
            </CalculatorFilter>
            Worst9mer name="worst9mer" rmsd_lookup_threshold="1.1"  only_helices="false" confidence="0" />
            <Holes name="holes_around_lig" threshold="-0.5" residue_selector="interface" normalize_per_atom="true" exclude_bb_atoms="true" confidence="0"/>
            <CavityVolume name="cavity" confidence="0"/>
            <BuriedUnsatHbonds name="buns_all_heavy_ball" report_all_heavy_atom_unsats="true" scorefxn="sfxn" cutoff="5" residue_surface_cutoff="20.0" ignore_surface_res="true" print_out_info_to_pdb="true" dalphaball_sasa="1" probe_radius="1.1" confidence="0" />
            <BuriedUnsatHbonds name="buns_bb_heavy_ball"  report_bb_heavy_atom_unsats="true"  scorefxn="sfxn" cutoff="5" residue_surface_cutoff="20.0" ignore_surface_res="true" print_out_info_to_pdb="true" dalphaball_sasa="1" probe_radius="1.1" confidence="0" />
            <BuriedUnsatHbonds name="buns_sc_heavy_ball"  report_sc_heavy_atom_unsats="true"  scorefxn="sfxn" cutoff="5" residue_surface_cutoff="20.0" ignore_surface_res="true" print_out_info_to_pdb="true" dalphaball_sasa="1" probe_radius="1.1" confidence="0" /> 
            <BuriedUnsatHbonds name="vbuns_all_heavy" use_reporter_behavior="true" report_all_heavy_atom_unsats="true" scorefxn="sfxn" ignore_surface_res="false" print_out_info_to_pdb="true" atomic_depth_selection="5.5" burial_cutoff="1000" confidence="0" />
            <BuriedUnsatHbonds name="sbuns_all_heavy" use_reporter_behavior="true" report_all_heavy_atom_unsats="true" scorefxn="sfxn" cutoff="4" residue_surface_cutoff="20.0" ignore_surface_res="true" print_out_info_to_pdb="true" dalphaball_sasa="1" probe_radius="1.1" atomic_depth_selection="5.5" atomic_depth_deeper_than="false" confidence="0" />
            <MoveBeforeFilter name="vbuns_all_heavy_no_ligand" mover="chain1only" filter="vbuns_all_heavy" confidence="0" />
            <MoveBeforeFilter name="sbuns_all_heavy_no_ligand" mover="chain1only" filter="sbuns_all_heavy" confidence="0" />
            <DSasa name="dsasa" lower_threshold="0" upper_threshold="1"/> 
            <ShapeComplementarity name="interface_sc" verbose="0" min_sc="0.55" write_int_area="1" write_median_dist="1" jump="1" confidence="0"/>

            <Time name="timed"/>
          </FILTERS>
          
        <SIMPLE_METRICS>
            <SapScoreMetric name="sap" />
            <SapScoreMetric name="sap_A"
                score_selector="chainA"
                sap_calculate_selector="chainA" sasa_selector="chainA" />
            <SecondaryStructureMetric name="dssp_string" />

        </SIMPLE_METRICS>

        <MOVERS>
            /turn on and off for test
            <DumpPdb name="after_fd1" fname="after_fd1_noFD.pdb" tag_time="1" scorefxn="sfxn" />
            <DumpPdb name="after_fd2" fname="after_fd2_noFD.pdb" tag_time="1" scorefxn="sfxn" />
            <DumpPdb name="after_relax" fname="after_relax_noFD.pdb" tag_time="1" scorefxn="sfxn" />
            <DumpPdb name="after_rmsd" fname="after_rmsd_noFD.pdb" tag_time="1" scorefxn="sfxn" />
        </MOVERS>
          
          
        <PROTOCOLS>
          <Add filter_name="timed"/>
          Add mover="FSP"/>
          Add filter="is_target_hbond_maintained"/>
          Add mover="install_protonated_his"/>
          Add mover="short_repack_and_min"/>
          Add filter="is_target_hbond_maintained"/>
          Add filter="ddg1"/>
          <Add mover="label_core"/>
          <Add mover="add_bb_cst"/>

          # turn off for only FR the MPNN model+ligand
          Add mover="fastDesign_stage1"/> 
          Add mover="after_fd1"/>
          Add mover="fastDesign_stage2"/> 
          Add mover="after_fd2"/>
          Add mover="save_pose"/>    
          <Add mover="fastRelax"/>
          Add mover="after_relax"/>
          Add filter="move_then_lig_rmsd"/> 
          Add mover="after_rmsd"/>
          <Add mover="rm_bb_cst"/>
                 
          <Add filter="score_per_res"/>
          <Add filter="geometry"/>

          <Add filter="contact_molecular_surface"/>
          <Add filter="ddg2"/>
          <Add filter="interface_buried_sasa"/>
          <Add filter="SC"/>
          <Add filter="holes_around_lig"/>
          <Add filter="nMET"/>
          <Add filter="nALA"/>
          <Add filter="nARG"/>
          <Add filter="nHIS"/>
          Add filter="ala_bdry_count"/>
          <Add filter="ala_core_count"/>
          Add filter="ala_loop_count"/>
          <Add filter="hb_lr_bb"/>
          <Add filter="hb_lr_bb_per_res"/>
          <Add filter="hb_sr_bb"/>
          <Add filter="hb_sr_bb_per_res"/>
          Add filter="worst9mer"/>
          <Add filter="hole"/>
          <Add filter="cavity"/>

          {protocols}
                    
          <Add filter="vbuns_all_heavy"/>
          <Add filter="sbuns_all_heavy"/>
          <Add filter="buns_all_heavy_ball"/>
          <Add filter="buns_bb_heavy_ball"/>
          <Add filter="buns_sc_heavy_ball"/>
          <Add filter="vbuns_all_heavy_no_ligand"/>
          <Add filter="sbuns_all_heavy_no_ligand"/>

          <Add filter="hydrophobic_residue_contacts"/>
          Add filter="mismatch_probability"/> /remove for now till fixed apptainer
          Add filter="sspred_overall"/>

          <Add filter="dsasa" /> /measure ligand burial rate, 0 is totally
          <Add filter="interface_sc"/>
          <Add metrics="sap_A" labels="sap_A"/>
          <Add metrics="sap" labels="sap_all"/>
          <Add filter_name="timed"/>

        </PROTOCOLS>
        <OUTPUT scorefxn="sfxn" /> /LA: this is the only way to export score to PDB file
    </ROSETTASCRIPTS>
    """
    #========================================================================
    # Build option parser
    #========================================================================
    outname = f"{tag}{SUFFIX}"
    print(f'outname {outname}')
    print('generated xml')
    key_contacts = get_all_close_res(pose, ligand_res_number)
    print(f'freese {key_contacts}')
    Add_key_contacts_to_rifres = f"""//////////Add key contacts////////
    <Index name="key_residues" resnums="{key_contacts}"/>
    <And name="dock_rifres_res" selectors="all_rifres_res,not_AVLI" />
    <Or name="rifres_res" selectors="dock_rifres_res,key_residues" />
        ////////////////
        """
    xml = xml.replace('<And name="rifres_res" selectors="all_rifres_res,not_AVLI" />', Add_key_contacts_to_rifres)
    #objs = protocols.rosetta_scripts.XmlObjects.create_from_string(xml)
    # CN's method
    design_task = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
    print('get xml')
    design_task.setup()
    print('setup xml')
    # anther way to set up xml
    # objs2 = protocols.rosetta_scripts.XmlObjects.create_from_string(xml)

    cst_list = get_all_atom_close_csts( pose, ligand_res_number, bb_only=False, sd=1.0, no_ligand_cst=True)
    for cst in cst_list:
        pose.add_constraint(cst)
    print('added cst')

    designed_pose = design_task.apply(pose)
    print(f'finished design')

    # designed_pose.dump_pdb(f'final.pdb')

    t1 = time.time()
    print("Design took ", t1-t0)

    #score again LA: for some reason I don't know, scores are not returned
    # scorefxn = get_fa_scorefxn('beta')
    # print(scorefxn.show(designed_pose))

    # try:
    #     df_scores = pandas.DataFrame.from_records([designed_pose.scores])
    #     df_scores.to_csv(f'{outname}.sc')
    # except:
    #     print("[ERROR] cannot save scores")
    # #     #return

    if args.save_to_pdb:
        print('save to pdb!')
        designed_pose.dump_pdb(f'{outname}.pdb')
        print(f'saved successfully')
        #to_pdb_file(designed_pose, outname+'.pdb')  #LA: do not deposite pdb for predictor. 210409: But for extend, deposite pdb
        #get scores
        score_df = get_total_scores(f'{outname}.pdb')
        score_df.to_csv(f'{outname}.sc')
    elif args.save_to_silent:
        silent_out = f'{outname}.silent'
        sfd_out = core.io.silent.SilentFileData(silent_out, False, False, "binary", core.io.silent.SilentFileOptions())
        add2silent(designed_pose,outname,sdf_out)
    elif args.save_score_only:
        score_df = get_total_scores(f'{outname}.pdb')
        score_df.to_csv(f'{outname}.sc')
        os.remove(f'{outname}.pdb')

    # End Checkpointing Functions



def main( tag, silent_structure, sfd_in, sfd_out ):
    """
    tag - the silent file tag
    silent_structure - the pose in silent format
    """
    
    t0 = time.time()
    print( "Attempting pose: %s"%tag )
    
    # Load pose from PDBLite
    pose = Pose()
    sfd_in.get_structure( tag ).fill_pose( pose )
    print('Pose loaded!')
    # pose.dump_pdb('test.pdb')
    # print('dump correct')
    print(f'tag {tag}')
    # print(f'silent_structure {silent_structure}')
    # print(f'sfd_out {sfd_out}')
    design( pose, tag, sfd_out )

    seconds = int(time.time() - t0)

    print( f"protocols.jd2.JobDistributor: {tag} reported success, generated in {seconds} seconds" )

#################################
# Begin Main Loop
#################################
if __name__ == '__main__':

    print(f'main version 17:08')

    if silent:

        silent_index = silent_tools.get_silent_index(silent)

        #sfd_out = None
        #if not lite: sfd_out = core.io.silent.SilentFileData(silent_out, False, False, "binary", core.io.silent.SilentFileOptions())

        sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
        sfd_in.read_file(silent)

        outname = os.path.basename(silent).replace('.silent','')
        silent_out = f'{outname}_designed.silent'
        sfd_out = rosetta.core.io.silent.SilentFileData(silent_out, False, False, "binary", core.io.silent.SilentFileOptions())

        # LA do this later
        checkpoint_filename = "check.point"
        debug = False

        # finished_structs = determine_finished_structs( checkpoint_filename )
        finished_structs = []

        for tag in tags:   
            if tag in finished_structs: continue

            silent_structure = silent_tools.get_silent_structure( silent, silent_index, tag )

            if debug: continue #main( tag, silent_structure, sfd_in, sfd_out ) #LA need to figure out why this is good

            else: # When not in debug mode the script will continue to run even when some poses fail
                t0 = time.time()

                try: main( tag, silent_structure, sfd_in, sfd_out )
                except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

                # except:
                #     seconds = int(time.time() - t0)
                #     print( "[ERROR] protocols.jd2.JobDistributor: %s failed in %i seconds with error: %s"%( tag, seconds, sys.exc_info()[0] ) )
                #     continue

            # LA do this later
            ## We are done with one pdb, record that we finished
            #record_checkpoint( pdb, checkpoint_filename )
    elif in_pdb:
        for tag in tags:
            t0 = time.time()
            silent_out = f'{tag}_designed.silent'
            sfd_out = rosetta.core.io.silent.SilentFileData(silent_out, False, False, "binary", core.io.silent.SilentFileOptions())
            pose = pose_from_file(in_pdb)
            print('Pose loaded!')
            # pose.dump_pdb('test.pdb')
            # print('dump correct')
            print(f'tag {tag}')
            # print(f'silent_structure {silent_structure}')
            # print(f'sfd_out {sfd_out}')
            design( pose, tag, sfd_out )

            seconds = int(time.time() - t0)

            print( f"protocols.jd2.JobDistributor: {tag} reported success, generated in {seconds} seconds" )

