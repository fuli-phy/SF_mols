#!/bin/bash
#SBATCH --partition=gpu4090
#SBATCH --job-name=F.t_cut3
#SBATCH --output=train.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1


python3 -u  model.py > log
