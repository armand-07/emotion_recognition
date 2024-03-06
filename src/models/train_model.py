from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import yaml
import os
import time
import shutil
import hashlib
from PIL import Image



import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torcheval.metrics.functional import multiclass_f1_score
import torchvision.transforms.v2 as transforms

from codecarbon import EmissionsTracker
from sklearn.metrics import confusion_matrix
import wandb

import numpy as np
import random

from src import PROCESSED_AFFECTNET_DIR, NUMBER_OF_EMOT, MODELS_DIR, AFFECTNET_CAT_EMOT
from src.data.dataset import AffectNetDataset
from src.models import archirectures as arch
from src.visualization import visualize as vis

from config import wandbAPIkey
import json


def data_loading(params):
    # Convert the image to float32 and scale it to [0,1]
    train_transforms = [transforms.ToDtype(torch.float32, scale = True)]
    val_transforms = [transforms.ToDtype(torch.float32, scale = True)]
    if params["data_augmentation"].lower() == "none":
        # No data augmentation, so the transforms remain unchanged
        pass
    else: 
        raise ValueError(f"Invalid data_augmentation parameter: {params['data_augmentation']}")
    if params["image_norm"].lower() == "imagenet":      # Normalize the image with the mean and std of the ImageNet dataset
        train_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], inplace = True))
        val_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], inplace = True))
    elif params["image_norm"].lower() == "affectnet":   # Normalize the image with the mean and std of the AffectNet dataset
        normalization_values = torch.load(
            os.path.join (PROCESSED_AFFECTNET_DIR, 'dataset_normalization_values.pt'))
        train_transforms.append(transforms.Normalize(mean=normalization_values['mean'], 
            std=normalization_values['std'], inplace = True))
        val_transforms.append(transforms.Normalize(mean=normalization_values['mean'], 
            std=normalization_values['std'], inplace = True))
    elif params["image_norm"].lower() == "none":
        pass
    else:
        raise ValueError(f"Invalid image_norm parameter: {params['image_norm']}")
    
    # Compose all concatenated transformations
    train_transforms = transforms.Compose(train_transforms)
    val_transforms = transforms.Compose(val_transforms)

    # Create the datasets
    dataset_train = AffectNetDataset(annotations_path=os.path.join(PROCESSED_AFFECTNET_DIR, "train.pkl"),
                                    img_transforms=train_transforms)
    dataset_val = AffectNetDataset(annotations_path=os.path.join(PROCESSED_AFFECTNET_DIR, "val.pkl"),
                                    img_transforms=val_transforms)
    # Load weights
    train_weights = torch.load(os.path.join(PROCESSED_AFFECTNET_DIR, "data_weights_train.pt"))
    val_weights = torch.load(os.path.join(PROCESSED_AFFECTNET_DIR, "data_weights_val.pt"))
    if params['epoch_samples'] == None:
        epoch_train_size = len(dataset_train)
        epoch_val_size = len(dataset_val)
    else:
        epoch_train_size = params['epoch_samples']
        epoch_val_size = params['epoch_samples']
    # Load sampler
    sampler_train = WeightedRandomSampler(train_weights,epoch_train_size, replacement=True)
    sampler_val = WeightedRandomSampler(val_weights, epoch_val_size, replacement=True)
    # Create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=params['batch_size'], 
                                pin_memory=True, sampler=sampler_train, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=params['batch_size'], 
                                pin_memory=True, sampler=sampler_val, drop_last=True)
    return dataloader_train, dataloader_val


def seed_everything(seed):
    """Set seeds to allow reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def model_creation(params):
    seed_everything(params['random_seed'])
    
    if params['arch'].lower() == "resnet50":
        model = arch.resnet50(pretrained = True)
    elif params['arch'].lower() == "resnext50_32x4d":
        model = arch.resnext50_32x4d(pretrained = True)

    # Define criterion
    criterion = nn.CrossEntropyLoss(reduction = 'mean') # Note that this case is equivalent to the combination of LogSoftmax and NLLLoss.
    # Define optimizer
    if params["optimizer"].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=float(params["lr"]))
    else:
        optimizer = optim.SGD(model.parameters(), lr=float(params["lr"]), momentum=params["momentum"])

    # We need GPU to train the model
    assert torch.cuda.is_available()
    print(f'Using CUDA with {torch.cuda.device_count()} GPUs')
    print(f'Using CUDA device:{torch.cuda.get_device_name(torch.cuda.current_device())}')

    # Move model to GPU
    device = torch.device("cuda")
    model.to(device)

    return model, criterion, optimizer, device


def train(
        train_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn, 
        optimizer: torch.optim, 
        device: torch.device,
        epoch: int,
        params: dict
        ) -> Tuple[np.float32, np.float32]:

    # Switch to train mode
    model.train()
    # I will save the values of the accuracies in this list to return the mean of the whole dataset at the end
    f1_scores = torch.empty(len(train_loader), dtype=torch.float32, device = 'cpu')
    global_epoch_loss = 0.0
    
    for i, (imgs, cat_target, cont_target) in tqdm(enumerate(train_loader), total=len(train_loader), desc = f'Epoch{epoch+1}: TRAIN'):
        # Reset gradients
        optimizer.zero_grad()
        # Move images and target to gpu
        imgs = imgs.to(device)
        cat_target = cat_target.to(device)

        # Forward batch of images through the network
        prediction = model(imgs)
        # Compute the loss
        loss = criterion(prediction, cat_target)
        # Compute gradient and optimize weights
        loss.backward()
        optimizer.step()

        # Measure F1-score
        f1_scores[i] = multiclass_f1_score(input=prediction, target=cat_target, num_classes=NUMBER_OF_EMOT).cpu()
        global_epoch_loss += loss.data.item()*params['batch_size'] # Accumulate the loss
        if i % params['step_log_interval'] == 0:
            tqdm.write(f'TRAIN [{i+1}/{len(train_loader)}] F1-score {f1_scores[i].item():.3f} Loss {loss.item():.3f}')
            #wandb.log({'Train loss step evolution': loss.item()}, step=epoch+(i+1/len(train_loader)))
            #wandb.log({'Train F1 Score step evolution': f1_scores[i].item()}, step=epoch+(i+1/len(train_loader)))
    return torch.mean(f1_scores).item(), global_epoch_loss/len(train_loader.dataset)

def validate(
        val_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn, 
        device: torch.device,
        epoch: int,
        params: dict
        ) -> np.float32:

    # Switch model to evaluate mode
    model.eval()

    # I will save the values of the accuracies in this list to return the mean of the whole dataset at the end
    f1_scores = torch.empty(len(val_loader), dtype=torch.float32)
    all_preds = torch.empty(0, device = 'cpu')
    all_targets = torch.empty(0, device = 'cpu')
    global_epoch_loss = 0.0

    with torch.no_grad():  #There is no need to compute gradients
        for i, (imgs, cat_target, cont_target) in tqdm(enumerate(val_loader), total=len(val_loader), desc = f'(VAL)Epoch {epoch+1}'):
            # Move images to gpu
            imgs = imgs.to(device)
            cat_target = cat_target.to(device)

            # Forward batch of images through the network
            prediction = model(imgs)
            # Compute the loss
            loss = criterion(prediction, cat_target)
            # Measure F1-score
            # Input (Tensor) – Tensor of label predictions. It could be the predicted labels, with shape of (n_sample, ). 
            # It could also be probabilities or logits with shape of (n_sample, n_class). torch.argmax will be used to convert input into predicted labels.
            # target (Tensor) – Tensor of ground truth labels with shape of (n_sample, ).
            f1_scores[i] = multiclass_f1_score(input=prediction, target=cat_target, num_classes=NUMBER_OF_EMOT).item()
            global_epoch_loss += loss.data.item()*imgs.shape[0] # Accumulate the loss

            # PREDS ARE IN LOGITS NO IN PROBABILITY
            all_preds = torch.cat((all_preds, torch.argmax(prediction, dim=1).cpu())) # Get the predicted labels using argmax
            all_targets = torch.cat((all_targets, cat_target.cpu()))
            
            if i % params['step_log_interval'] == 0:
                tqdm.write(f'VAL [{i+1}/{len(val_loader)}] F1-score {f1_scores[i].item():.3f} Loss {loss.item():.3f}')
                #wandb.log({'Val loss step evolution': loss.item()}, step=epoch+(i+1/len(val_loader)))
                #wandb.log({'Val F1 Score step evolution': f1_scores[i].item()}, step=epoch+(i+1/len(val_loader)))
    
    conf_matrix = confusion_matrix(all_targets.numpy(), all_preds.numpy(), normalize = 'true')
    chart = vis.create_conf_matrix(conf_matrix)
    wandb.log({'Confusion Matrix': chart}, step = epoch+1)

    return torch.mean(f1_scores).item(), global_epoch_loss / len(val_loader.dataset)


def save_checkpoint(
        state: 'dict', 
        is_best: bool, 
        ) -> None:
    torch.save(state, os.path.join(MODELS_DIR,'checkpoint.pt'))
    
    # Save an extra copy if it is the best model yet
    if is_best:
        shutil.copyfile(os.path.join(MODELS_DIR,'checkpoint.pt'), os.path.join(MODELS_DIR,'model_best.pt'))  


def main(params):
    # Clear the output directory
    if os.path.exists(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
        os.makedirs(MODELS_DIR)

    # Load the data
    print('Loading data...')
    dataloader_train, dataloader_val = data_loading(params)
    # Create and prepare the model and the optimizer
    print('Creating model and setting optimizer and criterion...')
    model, criterion, optimizer, device = model_creation(params)
    
    # Start measuring the emissions tracker
    tracker = EmissionsTracker(measure_power_secs=10, output_dir=MODELS_DIR,
                               log_level= "warning") # measure power every 10 seconds
    tracker.start()

    config = {
    "data_augmentation": "None",
    "image_norm": "ImageNet",
    "arch": "ResNet50", # architecture name 
    "batch_size": 256, # training and valid batch size
    "lr": 0.00001, # learning rate
    "momentum": 0.9, # SGD momentum, for SGD only
    "optimizer": 'adam', # optimization method: sgd | adam
    "epochs": 20,  # maximum number of epochs to train
    "patience": 5, # how many epochs of no loss improvement should we wait before stop training
    }

    # Convert the dictionary to a string
    config_string = json.dumps(config, sort_keys=True)

    # Get hash of the string representing the configuration
    num_run = hashlib.md5(config_string.encode()).hexdigest()

    print(f'WanDB run number hash id: {num_run}') # Print the run number hash id
    wandb.login(key=wandbAPIkey)
    wandb.init(
    entity='armand-07',
    project="TFG Facial Emotion Recognition",
    config=config,
    name=f'run_{num_run}',
    )
    #wandb.watch(model, log_freq=100) # use wandb.watch with a log_freq=100.
    # Define the training parameters
    best_f1_score = 0.0
    best_epoch = 0

    # Training with early stopping if patience threshold is surpassed
    t0 = time.time()

    for epoch in range(params['epochs']):
        #mean_f1_score_train, mean_loss_train  = train(dataloader_train, model, criterion, optimizer, device, epoch, params)
        #wandb.log({"Train F1-Score mean per epoch": mean_loss_train}, step=epoch+1)
        #wandb.log({"Train loss mean per epoch": mean_f1_score_train}, step=epoch+1)

        mean_f1_score_val, mean_loss_val = validate(dataloader_val, model, criterion, device, epoch, params)
        wandb.log({"Val F1-Score mean per epoch": mean_f1_score_val}, step=epoch+1)
        wandb.log({"Val loss mean per epoch": mean_loss_val}, step=epoch+1)

        # Remember best f1-score and save checkpoint
        if mean_f1_score_val > best_f1_score:
            is_best = True
            best_f1_score = mean_f1_score_val
            best_epoch = epoch

        save_checkpoint({
            'epoch': epoch+1,
            'arch': params['arch'],
            'state_dict': model.state_dict(),
            'best_f1_score': best_f1_score,
            'params': params,
        }, is_best)

        if epoch - best_epoch == params['patience']:
            print(f'Early stopping at epoch {epoch}')
            t1 = time.time()
            print(f'Training time: {t1-t0:.2f} seconds')
            break

    wandb.finish()


if __name__ == '__main__':
    # Path of the parameters file
    params_path = Path("params.yaml")
    # Read data preparation parameters
    with open(params_path, "r", encoding='utf-8') as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["training"]
        except yaml.YAMLError as exc:
            print(exc)
    main(params)