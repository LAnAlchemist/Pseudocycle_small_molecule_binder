#for RMSD
# from Derrick, he got from someone else

sys.path.append("/software/pyrosetta3.8/latest/")

from pyrosetta import *
from pyrosetta.rosetta import *




def fa_pose_from_str(pdb_str):
    pose = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose, pdb_str)
    return pose


def backbone_rmsd( move_pose, to_pose, atoms=["N", "CA", "C"] ):

    move_res = np.array(list(range(1, move_pose.size()+1)))

    to_res = np.array(list(range(1, to_pose.size()+1)))

    move_to_pairs = []
    coords_move = utility.vector1_numeric_xyzVector_double_t()
    coords_to = utility.vector1_numeric_xyzVector_double_t()

    for i in range(len(move_res)):
        seqpos_move = move_res[i]
        seqpos_to = to_res[i]

        move_to_pairs.append((seqpos_move, seqpos_to))

        for atom in atoms:
            coords_move.append(move_pose.residue(seqpos_move).xyz(atom))
            coords_to.append(to_pose.residue(seqpos_to).xyz(atom))


    move_pose_copy = move_pose.clone()


    backbone_rmsd = 0

    backbone_distances = []

    if ( len(move_to_pairs) > 0 ):

        rotation_matrix = numeric.xyzMatrix_double_t()
        move_com = numeric.xyzVector_double_t()
        ref_com = numeric.xyzVector_double_t()

        protocols.toolbox.superposition_transform( coords_move, coords_to, rotation_matrix, move_com, ref_com )

        protocols.toolbox.apply_superposition_transform( move_pose, rotation_matrix, move_com, ref_com )

        for seqpos_move, seqpos_to in move_to_pairs:
            for atom in atoms:
                backbone_distance = move_pose.residue(seqpos_move).xyz(atom).distance_squared(to_pose.residue(seqpos_to).xyz(atom))
                backbone_rmsd += backbone_distance
                backbone_distances.append(backbone_distance)

        backbone_rmsd /= len(backbone_distances)
        backbone_rmsd = np.sqrt(backbone_rmsd)

    print(f"backbone rmsd for residues {move_res}: {backbone_rmsd}")
    return backbone_rmsd

def pair_RMSD( pose, other_pose):

    if 1:
    #try:
        bb_rmsd = backbone_rmsd( pose, other_pose)
        return bb_rmsd
    if not 1:
    #except:
        print("rmsd problem; likley colinear atoms")
        return 999




#pose is your input pose that was used to generate all the sequences being tested
#pdbstr comes from the af2 output and usually used to write pdb to disk
bb_rmsd = pair_RMSD(pose, fa_pose_from_str(pdbstr))

if bb_rmsd < some_threshold:
#dump pdb