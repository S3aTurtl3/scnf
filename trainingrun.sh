#!/bin/bash
#SBATCH -J scnf # Job name
#SBATCH -p mit_normal_gpu # Partition(s) (separate with
# commas if using multiple)
#SBATCH -c 32 # Number of cores
#SBATCH -t 0-01:30:00 # Time (D-HH:MM:SS)
#SBATCH --mem=40G # Memory
#SBATCH -o scnfpy_%j.o # Name of standard output file
#SBATCH -e scnfpy_%j.e # Name of standard error file

# load software environment
module load miniforge
# print a statement
cd /home/orealao/orcd/pool/scnf # go to home directory
# execute python code
pixi run python cnf_siren_ot.py