#!/bin/bash
#SBATCH --nodelist=gpic09
#SBATCH -c 1
#SBATCH --mem 16G
#SBATCH --gres=gpu:1
#SBATCH --time 16:00:00 
python -m src.models.predict_model --mode img --file "test3.jpg" --wandb_id iconic-sweep-19

