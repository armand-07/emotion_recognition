#!/bin/bash
#SBATCH --nodelist=gpic10
#SBATCH -c 2
#SBATCH --mem 64G
#SBATCH --gres=gpu:1
#SBATCH --time 10:00:00 
python -m src.data.make_processed