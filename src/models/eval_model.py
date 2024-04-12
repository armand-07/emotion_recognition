from pathlib import Path
import yaml
import os
import argparse

import torch
from torch import nn

import wandb

import numpy as np
import random

from src import PROCESSED_AFFECTNET_DIR, NUMBER_OF_EMOT, MODELS_DIR, AFFECTNET_CAT_EMOT
from src.models import architectures as arch
from src.models.train_model import validate
from src.data.dataset import create_dataloader
from src.models.POSTER_V2.main import *

from config import wandbAPIkey



def main(wandb_id):   
    wandb.login(key=wandbAPIkey)
    run = wandb.init(
    entity='armand-07',
    project="TFG Facial Emotion Recognition",
    job_type="test",
    )
    run_name = wandb.run.name
    print(f'WanDB run name is: {run_name}') # Print the run number hash id
    
    if wandb_id == None: # If no wandb_id is provided, use the params.yaml file
        # Path of the parameters file
        params_path = Path("params.yaml")
        # Read data preparation parameters
        with open(params_path, "r", encoding='utf-8') as params_file:
            try:
                params = yaml.safe_load(params_file)
                params = params["eval"]
            except yaml.YAMLError as exc:
                print(exc)
    else: # If a wandb_id is provided, download the model weights
        artifact_dir = arch.get_wandb_artifact(wandb_id, run)
        local_artifact = torch.load(os.path.join(artifact_dir, "model_best.pt"))
        params = local_artifact["params"]

    wandb.config.update(params)

    batch_size = 32
    # Load the data
    print('Loading data...')
    arch.seed_everything(params['random_seed'])
    dataloader_test = create_dataloader(datasplit = "test", batch_size = batch_size, 
                                        image_norm = params['image_norm'], num_workers = 2)
    # Create and prepare the model and the optimizer
    print('Creating model and setting optimizer and criterion...')
    if wandb_id == None:
        model, device = arch.model_creation(params['arch'], params['weights'])
    else:
        model, device = arch.model_creation(params['arch'], local_artifact['state_dict'])

    criterion = nn.CrossEntropyLoss(reduction = 'mean') # Note that this case is equivalent to the combination of LogSoftmax and NLLLoss.

    metrics = validate(dataloader_test, model, criterion, device, 0, batch_size, run)
    print(metrics)

    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_id', type=str, default=None, help='Run id to take the model weights')
    return parser.parse_args() 


if __name__ == '__main__':
    args = parse_args()
    main(args.wandb_id)