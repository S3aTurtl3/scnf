#!/bin/bash
#SBATCH -J minisi # Job name
#SBATCH -p mit_normal_gpu # Partition(s) (separate with
# commas if using multiple)
#SBATCH -c 11 # Number of cores
#SBATCH -t 0-01:30:00 # Time (D-HH:MM:SS)
#SBATCH --mem=40G # Memory
#SBATCH -o minisi_%j.o # Name of standard output file
#SBATCH -e minisi_%j.e # Name of standard error file
#SBATCH --mail-user=orealao@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# load software environment
module load miniforge
# print a statement
cd /home/orealao/orcd/pool/scnf # go to home directory
# execute python code
pixi run python cnfsmini.py --batch-size=250 --trials=10