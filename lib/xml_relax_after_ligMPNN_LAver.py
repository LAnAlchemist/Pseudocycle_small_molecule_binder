XML_BSITE_REPACK_MIN = """
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="scorefxn_full" weights="beta_genpot">
      <Reweight scoretype="coordinate_constraint" weight="0.5"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.1" />
    </ScoreFunction>
    <ScoreFunction name="scorefxn_soft" weights="beta_genpot_soft">
      <Reweight scoretype="coordinate_constraint" weight="1.0"/>
      <Reweight scoretype="fa_rep" weight="0.2"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.1" />
    </ScoreFunction>
  </SCOREFXNS>

  <RESIDUE_SELECTORS>
    <Index name="repack_res" resnums="{0}"/>
    <Not name="fix_res" selector="repack_res"/>
    <Chain name="chB" chains="B"/>
  </RESIDUE_SELECTORS>

  <TASKOPERATIONS>
    <OperateOnResidueSubset name="dont_repack" selector="fix_res">
      <PreventRepackingRLT/>
    </OperateOnResidueSubset>
    <OperateOnResidueSubset name="repack_selected_res" selector="repack_res">
      <RestrictToRepackingRLT/>
    </OperateOnResidueSubset>
    <InitializeFromCommandline name="init"/>
    <ExtraRotamersGeneric name="ex1ex2" ex1="1" ex2="1"/>
  </TASKOPERATIONS>

  <FILTERS>
    <ScoreType name="totalscore" scorefxn="scorefxn_full" threshold="9999" confidence="0"/>
    <ResidueCount name="nres" confidence="0" />
    <CalculatorFilter name="res_totalscore" confidence="0" equation="SCORE/NRES" threshold="999">
      <Var name="SCORE" filter_name="totalscore" />
      <Var name="NRES" filter_name="nres" />
    </CalculatorFilter>
    <Ddg name="ddg" threshold="9999999" jump="1" repeats="1" repack="1" repack_bound="true" repack_unbound="true" relax_unbound="false" scorefxn="scorefxn_full"/>
    ContactMolecularSurface name="cms" target_selector="chB" binder_selector="repack_res"/>
    <ContactMolecularSurface name="cms" target_selector="chB" binder_selector="repack_res" use_rosetta_radii="true"/>
    {1}
  </FILTERS>

  <MOVERS>
    <AddConstraints name="add_ca_csts" >
      <CoordinateConstraintGenerator name="coord_cst_gen" ca_only="true"/>
    </AddConstraints>
    ConstraintSetMover name="add_lig_d_csts" add_constraints="1" cst_file="1" />
    ConstraintSetMover name="rm_lig_d_csts" cst_file="none" />
    <ClearConstraintsMover name="rm_all_cst" />
    <RemoveConstraints name="rm_csts" constraint_generators="coord_cst_gen"/>
    <PackRotamersMover name="repack" scorefxn="scorefxn_full" task_operations="init,ex1ex2,repack_selected_res,dont_repack"/>
    <PackRotamersMover name="repack_soft" scorefxn="scorefxn_soft" task_operations="init,ex1ex2,repack_selected_res,dont_repack"/>
     MinMover name="min_full" bb="1" chi="1" jump="0" scorefxn="scorefxn_full"/>
    <MinMover name="min_full" bb="1" chi="1" jump="ALL" scorefxn="scorefxn_full"/>
    <MinMover name="min_soft" bb="1" chi="1" jump="ALL" scorefxn="scorefxn_soft"/>
  </MOVERS>

  <PROTOCOLS>
    Add mover="add_ca_csts"/>
    Add mover="add_lig_d_csts"/>
    <Add mover="repack_soft"/>
    <Add mover="min_soft"/>
    <Add mover="repack"/>
    <Add mover="min_full"/>
    Add mover="rm_csts"/>
    Add mover="rm_lig_d_csts"/>
    <Add mover="rm_all_cst"/>
    {2}
    <Add filter="ddg"/>
    <Add filter="cms"/>
    <Add filter="res_totalscore"/>
    <Add filter="totalscore"/>
  </PROTOCOLS>

</ROSETTASCRIPTS>

"""

XML_BSITE_FASTRELAX = """
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="scorefxn_full" weights="beta_genpot">
      <Reweight scoretype="coordinate_constraint" weight="0.5"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.1" />
    </ScoreFunction>
    <ScoreFunction name="scorefxn_soft" weights="beta_genpot_soft">
      <Reweight scoretype="coordinate_constraint" weight="1.0"/>
      <Reweight scoretype="fa_rep" weight="0.2"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.1" />
    </ScoreFunction>
  </SCOREFXNS>

  <RESIDUE_SELECTORS>
    <Index name="repack_res" resnums="{0}"/>
    <Not name="fix_res" selector="repack_res"/>
    <Chain name="chB" chains="B"/>
  </RESIDUE_SELECTORS>

  <TASKOPERATIONS>
    <OperateOnResidueSubset name="dont_repack" selector="fix_res">
      <PreventRepackingRLT/>
    </OperateOnResidueSubset>
    <OperateOnResidueSubset name="repack_selected_res" selector="repack_res">
      <RestrictToRepackingRLT/>
    </OperateOnResidueSubset>
    #
    <InitializeFromCommandline name="init"/>
    <ExtraRotamersGeneric name="ex1ex2" ex1="1" ex2="1"/>
  </TASKOPERATIONS>

  <FILTERS>
    <ScoreType name="totalscore" scorefxn="scorefxn_full" threshold="9999" confidence="0"/>
    <ResidueCount name="nres" confidence="0" />
    <CalculatorFilter name="res_totalscore" confidence="0" equation="SCORE/NRES" threshold="999">
      <Var name="SCORE" filter_name="totalscore" />
      <Var name="NRES" filter_name="nres" />
    </CalculatorFilter>

    <Ddg name="ddg" threshold="99999" jump="1" repeats="1" repack="1" repack_bound="true" repack_unbound="true" relax_unbound="false" scorefxn="scorefxn_full"/>
    ContactMolecularSurface name="cms" target_selector="chB" binder_selector="repack_res"/>
    <ContactMolecularSurface name="cms" target_selector="chB" binder_selector="repack_res" use_rosetta_radii="true"/>
    {1}
  </FILTERS>

  <MOVERS>
    <AddConstraints name="add_ca_csts" >
      <CoordinateConstraintGenerator name="coord_cst_gen" ca_only="true"/>
    </AddConstraints>
    ConstraintSetMover name="add_lig_d_csts" add_constraints="1" cst_file="1"/>
    <ConstraintSetMover name="rm_lig_d_csts" cst_file="none"/>
    <ClearConstraintsMover name="rm_all_cst" />
    <RemoveConstraints name="rm_csts" constraint_generators="coord_cst_gen"/>

    <FastRelax name="fastrelax" scorefxn="scorefxn_full" task_operations="init,ex1ex2,repack_selected_res,dont_repack" repeats="1" /> #cst_file="1" 
  </MOVERS>

  <PROTOCOLS>
    <Add mover="add_ca_csts"/>
    Add mover="add_lig_d_csts"/>
    <Add mover="fastrelax"/>
    Add mover="rm_csts"/>
    Add mover="rm_lig_d_csts"/>
    <Add mover="rm_all_cst"/>
    {2}
    <Add filter="ddg"/>
    <Add filter="cms"/>
    <Add filter="res_totalscore"/>
    <Add filter="totalscore"/>
  </PROTOCOLS>

</ROSETTASCRIPTS>

"""
