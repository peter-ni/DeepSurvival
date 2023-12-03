#!/bin/bash


### Set Job Name
#SBATCH --job-name=DeepCNA

### -n requests total cores. In this case, we are requesting 2 cores
#SBATCH -n 2
### -N requests nodes. In this case, we are requesting 1 node
#SBATCH -N 1

### Job/Output/Error File Names
#SBATCH --output job%j.out
#SBATCH --error job%j.err


module load python3/anaconda/2023.9

pip install jax
pip install jaxlib


python3 DeepCNA-Net/hpc_test_file.py



echo "Current Date"
date