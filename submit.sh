#!/bin/bash
#
#SBATCH --job-name=GPUtest
#SBATCH --time=850:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --partition=kipac
##SBATCH --gres gpu:2

ml python/3.6.1
ml py-scipy/1.1.0_py36
ml py-numpy/1.17.2_py36

srun python3 test.py
