#!/bin/bash
#SBATCH --nodelist=gpic14
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00 
python -m src.models.train_model --mode extensive_train --wandb_id divine-sweep-14
