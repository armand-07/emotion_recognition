#!/bin/bash
#SBATCH -c 2
#SBATCH --nodelist=gpic14
#SBATCH --mem 32G
#SBATCH --gres=gpu:1
#SBATCH --time 0:15:00 
python -m src.models.eval_model --wandb_id cerulean-sweep-23


