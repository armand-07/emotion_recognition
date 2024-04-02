#!/bin/bash
#SBATCH --nodelist=gpic11
#SBATCH -c 2
#SBATCH --mem 32G
#SBATCH --gres=gpu:1
#SBATCH --time 16:00:00 
python -m src.models.eval_model --run_id hearty-sweep-25