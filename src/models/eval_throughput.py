import os
import argparse

import torch


import wandb

from src.models import architectures as arch
from src.data.dataset import create_dataloader
from src.models.metrics import eval_throughput
from src.models.POSTER_V2.main import *
from src.models import architectures_video as arch_v

from config import wandbAPIkey



def main(wandb_id:str = None) -> None:
    """ Main function to evaluate the model. If a wandb_id is provided, the model weights are downloaded from 
    the Weights and Biases server. If not, the parameters are read from the params.yaml file. The model is then
    loaded and the test set is evaluated. If distillation is enabled, the model is evaluated using the three
    different embedding methods. The results are logged to the Weights and Biases server.
    Params:
        - wandb_id (str): The id of the Weights and Biases run to download the model weights.
    Returns:
        - None
    """
    # Load the emotion model
    #wandb.login(key=wandbAPIkey)
    #api = wandb.Api()
    #artifact_dir = arch.get_wandb_artifact(wandb_id, api = api)
    #local_artifact = torch.load(os.path.join(artifact_dir, "model_best.pt"))

    #params = local_artifact["params"]

    #_, model, _, _, device = arch_v.load_video_models(wandb_id, 'nano', False, False)
    #_, model_cpu, _, _, device_cpu = arch_v.load_video_models(wandb_id, 'nano', False, True)

    model, device = arch.model_creation("poster", weights = 'affectnet_cat_emot')
    model_cpu, device_cpu = arch.model_creation("poster", weights = 'affectnet_cat_emot', device = 'cpu')
    batch_size = 9
    # Load the data
    print('Loading data...')
    arch.seed_everything(33)
    dataloader_test = create_dataloader(datasplit = "test", batch_size = batch_size, 
                                        image_norm = "affectnet", num_workers = 2)

    # Evaluate the model    
    print('Evaluating model...')
    eval_throughput(dataloader_test, model, device, batch_size)
    eval_throughput(dataloader_test, model_cpu, device_cpu, batch_size)




def parse_args():
    """Parse the arguments of the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_id', type=str, default=None, help='Run id to take the model weights')
    return parser.parse_args() 


if __name__ == '__main__':
    args = parse_args()
    main(args.wandb_id)