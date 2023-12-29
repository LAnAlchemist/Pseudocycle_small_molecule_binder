#!/bin/bash

#SBATCH -t 3:00:00
#SBATCH -n 10
#SBATCH -N1
#SBATCH --mem=60g
#SBATCH -o rifgen.log
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=linnaan

export OMP_NUM_THREADS=20

echo "started in "$(pwd) > ./rifgen.note
/home/bcov/rifdock/latest/rifgen @./rifgen.flags; rm -r cache

