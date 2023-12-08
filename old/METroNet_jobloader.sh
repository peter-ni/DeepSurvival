#!/bin/bash


### Set Job Name
#SBATCH --job-name=METroNet

### -n requests total cores. In this case, we are requesting 2 cores
#SBATCH -n 24
### -N requests nodes. In this case, we are requesting 1 node
#SBATCH -N 1

### Job/Output/Error File Names
#SBATCH --output job%j.out
#SBATCH --error job%j.err


module load python3/anaconda/2023.9

python3 pytorch_model.py



echo "Current Date"
date