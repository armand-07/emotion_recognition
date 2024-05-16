#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 16G
#SBATCH --nodelist=gpic14
#SBATCH --gres=gpu:1
#SBATCH --time 16:00:00 
python -m src.models.predict_model --mode save --input_path "test"