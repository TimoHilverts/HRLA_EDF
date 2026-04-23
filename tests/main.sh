#!/bin/bash
#SBATCH --job-name=Costs_SGD
#SBATCH --cpus-per-task=1 #matches 
#SBATCH --time=1:00:00
#SBATCH --mem=10G                 
#SBATCH --output=output/output_%j.txt
#SBATCH --error=output/error_%j.txt

module load Python/3.11.5-GCCcore-13.2.0
#python3 -m pip install scipy
# python3 -m pip install cma
python3 -m pip install .. --upgrade --quiet
python3 Costs_SGD.py