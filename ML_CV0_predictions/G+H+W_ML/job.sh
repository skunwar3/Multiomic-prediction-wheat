#!/bin/bash

#SBATCH --account=babar
#SBATCH --qos=babar-b
#SBATCH --output=status.%J.out
#SBATCH --error=status.%J.err
#SBATCH --job-name=TGW
#SBATCH --mail-type=All
#SBATCH --mail-user=skunwar@ufl.edu
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem=32gb
#SBATCH --distribution=cyclic:cyclic
#SBATCH -t 4-00:00:00
#SBATCH --err=error
#SBATCH --output=output
##SBATCH --dependency=afterok:8190473

module load python/3.8

python ML_genomic_model.py
