#!/bin/bash
#SBATCH --nodelist=gpic13
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH --gres=gpu:1
#SBATCH --time 16:00:00
python -m src.models.train_model --mode sweep --wandb_id wuj39ln3