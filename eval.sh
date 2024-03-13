#!/bin/bash
#SBATCH --nodelist=gpic11
#SBATCH -c 2
#SBATCH --mem 32G
#SBATCH --gres=gpu:1
#SBATCH --time 16:00:00 
python -m src.models.eval_model --mode sweep --run_id armand-07/TFG Facial Emotion Recognition/q21v65vf