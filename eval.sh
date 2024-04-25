#!/bin/bash
#SBATCH -c 2
#SBATCH --mem 32G
#SBATCH --gres=gpu:1
#SBATCH --time 1:00:00 
python -m src.models.eval_model --wandb_id iconic-sweep-19
