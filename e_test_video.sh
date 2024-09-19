#!/bin/bash
#SBATCH -c 1
#SBATCH --nodelist=gpic14
#SBATCH --mem 16G
#SBATCH --gres=gpu:1
#SBATCH --time 2:00:00 
python -m src.models.eval_model_video