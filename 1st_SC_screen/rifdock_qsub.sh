#!/bin/bash

#SBATCH -t 5:30:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=40G
#SBATCH -o log
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=linnaan

#export OMP_NUM_THREADS=20
#/home/norn/software/rifdock/build/apps/rosetta/rif_dock_test -scaffold_res $(cat POSFILE.list) -scaffolds $(cat SCAFFOLD.list) @rifdock.flags ; rm -r cache/
/mnt/home/bcov/rifdock/st/21_06_30_pssm/rifdock/build/apps/rosetta/rif_dock_test -scaffold_res $(cat POSFILE.list) -scaffolds $(cat SCAFFOLD.list) @rifdock.flags ; rm -r cache/
