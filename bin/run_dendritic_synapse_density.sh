#!/bin/bash -l
#SBATCH --job-name=syndensity
#SBATCH --time=24:00:00
#SBATCH --account=proj83
#SBATCH --partition=prod
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu

source /gpfs/bbp.cscs.ch/home/pokorny/BluePy2Kernel/bin/activate
python -u dendritic_synapse_density.py $1 $2 $3

# EXAMPLE HOW TO RUN: sbatch run_dendritic_synapse_density.sh CIRCUIT_CONFIG_PATH <#PARALLEL_JOBS, e.g. 72> <#DATA_SPLITS, e.g. 288>