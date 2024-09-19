#!/bin/bash
#SBATCH --nodelist=gpic12
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH --gres=gpu:1
#SBATCH --time 17:30:00
python -m src.models.train_model --mode sweep