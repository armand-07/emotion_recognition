from pathlib import Path
from tqdm import tqdm
import yaml
import os
import time
import shutil
import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torcheval.metrics.functional import multiclass_f1_score
from torcheval.metrics import MulticlassAccuracy
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from codecarbon import EmissionsTracker
from sklearn.metrics import confusion_matrix, top_k_accuracy_score, cohen_kappa_score
import wandb
from wandb.sdk import wandb_run

import numpy as np
import random

from src import PROCESSED_AFFECTNET_DIR, NUMBER_OF_EMOT, MODELS_DIR, AFFECTNET_CAT_EMOT
from src.data.dataset import AffectNetDataset
from src.models import architectures as arch
from src.models.metrics import compute_ROC_AUC_OVR
from src.visualization import visualize as vis

from config import wandbAPIkey



def data_transforms(params):
    # Convert the image to float32 and scale it to [0,1]
    train_transforms = []
    val_transforms = []
    if params["daug_randomhorizontalflip"]:
        train_transforms.append(A.HorizontalFlip(p=0.5))
    if params["daug_colorjitter"]:
        train_transforms.append(A.ColorJitter(brightness=[0.85, 1.15], 
            contrast=[0.9,1.1], saturation=[0.75,1.1], hue=[-0.01,0.02], 
            p = 0.5))
    if params["daug_randomaffine"]:
        train_transforms.append(A.ShiftScaleRotate(rotate_limit=(-15, 15), 
            shift_limit=(0, 0.1), scale_limit=(-0.1, 0.1), 
            border_mode = cv2.BORDER_CONSTANT, value = 0.0, p = 0.5))
    if params["daug_randomerasing"]:
        train_transforms.append(A.CoarseDropout(max_height=85, min_height = 16, 
            max_width = 85, min_width = 16, fill_value = 0.0, max_holes = 1, 
            min_holes = 1, p = 0.5))

    # Normalize the image
    if params["image_norm"].lower() == "imagenet":      # Normalize the image with the mean and std of the ImageNet dataset
        train_transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]))
        val_transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]))
    elif params["image_norm"].lower() == "affectnet":   # Normalize the image with the mean and std of the AffectNet dataset
        normalization_values = torch.load(
            os.path.join (PROCESSED_AFFECTNET_DIR, 'dataset_normalization_values.pt'))
        train_transforms.append(A.Normalize(mean=normalization_values['mean'], 
            std=normalization_values['std']))
        val_transforms.append(A.Normalize(mean=normalization_values['mean'], 
            std=normalization_values['std']))
    else:
        raise ValueError(f"Invalid image_norm parameter: {params['image_norm']}")
    
    train_transforms.append(ToTensorV2())
    val_transforms.append(ToTensorV2())
    
    # Convert the list of transforms to a Compose object
    return A.Compose(train_transforms), A.Compose(val_transforms)



def data_loading(params):
    train_transforms, val_transforms = data_transforms(params)

    # Create the datasets
    dataset_train = AffectNetDataset(annotations_path=os.path.join(PROCESSED_AFFECTNET_DIR, "train.pkl"),
                                    img_transforms=train_transforms)
    dataset_val = AffectNetDataset(annotations_path=os.path.join(PROCESSED_AFFECTNET_DIR, "val.pkl"),
                                    img_transforms=val_transforms)
    # Load weights
    train_weights = torch.load(os.path.join(PROCESSED_AFFECTNET_DIR, "data_weights_train.pt"))
    #val_weights = torch.load(os.path.join(PROCESSED_AFFECTNET_DIR, "data_weights_val.pt"))
    if params['epoch_samples'] == "original":
        epoch_train_size = len(dataset_train)
    else:
        epoch_train_size = params['epoch_samples']
    # Load sampler
    sampler_train = WeightedRandomSampler(train_weights,epoch_train_size, replacement=True)
    #sampler_val = WeightedRandomSampler(val_weights, 1000, replacement=True)
    # Create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=params['batch_size'], 
                                pin_memory=True, sampler=sampler_train, drop_last=True, num_workers=2)
    dataloader_val = DataLoader(dataset_val, batch_size=params['batch_size'], 
                                pin_memory=True, shuffle=True, drop_last=True, num_workers=2)
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
    # Create the model following the architecture specified in the parameters
    if params['arch'].lower() == "resnet50":
        pretrained = False
        if params['pretraining'].lower() == 'imagenet':
            pretrained = True
        model = arch.resnet50(pretrained = pretrained)
    elif params['arch'].lower() == "resnext50_32x4d":
        model = arch.resnext50_32x4d(pretrained = True)

    # We need GPU to train the model
    assert torch.cuda.is_available()
    print(f'Using CUDA with {torch.cuda.device_count()} GPUs')
    print(f'Using CUDA device:{torch.cuda.get_device_name(torch.cuda.current_device())}')
    # Move model to GPU
    device = torch.device("cuda")
    model.to(device)

    # Define criterion
    criterion = nn.CrossEntropyLoss(reduction = 'mean') # Note that this case is equivalent to the combination of LogSoftmax and NLLLoss.
    # Define optimizer
    if params["optimizer"].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=float(params["lr"]))
    elif params["optimizer"].lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=float(params["lr"]), momentum=float(params["momentum"]))
    elif params["optimizer"].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=float(params["lr"]), momentum=float(params["momentum"]))
    else:
        raise ValueError(f"Invalid optimizer parameter: {params['optimizer']}")
    
    return model, criterion, optimizer, device



def train(
        train_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn, 
        optimizer: torch.optim, 
        device: torch.device,
        epoch: int,
        params: dict,
        run: wandb_run.Run
        ) -> None:

    # Switch to train mode
    model.train()
    # I will save the values of the accuracies in this list to return the mean of the whole dataset at the end
    global_epoch_loss = torch.zeros(1, dtype=torch.float, device = device)
    acc = MulticlassAccuracy(device=device)
    
    for i, (imgs, cat_target, cont_target) in tqdm(enumerate(train_loader), 
                                                   total=len(train_loader), desc = f'(TRAIN)Epoch {epoch+1}', 
                                                   miniters=int(len(train_loader)/100)):
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
        # Measure accuracy and loss
        acc.update(prediction, cat_target)
        global_epoch_loss += loss
        if i % 100 == 0:
            predicted_label = torch.argmax(prediction, dim=1)
            acc_batch = torch.sum(predicted_label == cat_target).item()/params['batch_size']*100
            tqdm.write(f"TRAIN [{i+1}/{len(train_loader)}], Batch accuracy: {acc_batch:.3f}%; Batch Loss: {loss.item():.3f}")
            if torch.isnan(loss):
                raise ValueError(f"Loss is NaN at epoch {epoch+1} and step {i+1}, caused by gradient clipping or small learning rate")
    # Reset gradients
    optimizer.zero_grad()

    # Compute the metrics
    acc = acc.compute().item()
    global_epoch_loss = global_epoch_loss / (len(train_loader)) # all batches have same size
    run.log({"Train accuracy per epoch": acc}, step=epoch+1)
    run.log({"Train loss mean per epoch": global_epoch_loss}, step=epoch+1)



def validate(
        val_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn, 
        device: torch.device,
        epoch: int,
        params: dict,
        run: wandb_run.Run
        ) -> dict:

    # Switch model to evaluate mode
    model.eval()

    # I will save the values of the accuracies in this list to return the mean of the whole dataset at the end
    acc1 = torch.zeros(1, dtype=torch.int, device = device)
    all_preds_labels = torch.empty(0, device = 'cpu')
    all_preds_dist = torch.empty(0, device = 'cpu')
    softmax = nn.Softmax(dim=1)
    all_targets = torch.empty(0, device = 'cpu')
    global_epoch_loss = 0.0
    
    with torch.no_grad():  #There is no need to compute gradients
        for i, (imgs, cat_target, _) in tqdm(enumerate(val_loader), 
                                                       total=len(val_loader), desc = f'(VAL)Epoch {epoch+1}', 
                                                        miniters=int(len(val_loader)/50)):
            # Move images to gpu
            imgs = imgs.to(device)
            cat_target = cat_target.to(device)

            # Forward batch of images through the network
            prediction = model(imgs)
            predicted_label = torch.argmax(prediction, dim=1) # Get the predicted labels using argmax
            
            # Compute the loss
            loss = criterion(prediction, cat_target)
            # Measure metrics
            acc1 += torch.sum(predicted_label == cat_target)
            
            global_epoch_loss += loss.data.item()*imgs.shape[0] # Accumulate the loss

            # PREDS ARE IN LOGITS NO IN PROBABILITY
            all_preds_labels = torch.cat((all_preds_labels, predicted_label.cpu())) 
            all_preds_dist = torch.cat((all_preds_dist, softmax(prediction).cpu())) # Apply softmax to the predictions
            all_targets = torch.cat((all_targets, cat_target.cpu()))
            
            if i % 50 == 0:
                acc_batch = torch.sum(predicted_label == cat_target).item()/params['batch_size']*100
                tqdm.write(f'VAL [{i+1}/{len(val_loader)}], Batch accuracy: {acc_batch:.2f}%; Batch Loss: {loss.item():.3f}')

    # Compute the metrics
    acc1 = acc1.item()/(len(val_loader) * params['batch_size']) # Log the accuracy
    acc2 = top_k_accuracy_score(all_targets.numpy(), all_preds_dist.numpy(), k=2, normalize=True) # Log the top-2 accuracy
    global_epoch_loss = global_epoch_loss /(len(val_loader) * params['batch_size']) # Mean loss
    f1_score = multiclass_f1_score(input=all_preds_labels, target=all_targets, num_classes=NUMBER_OF_EMOT).item() # F1-Score
    cohen_kappa = cohen_kappa_score(all_targets, all_preds_labels) # Cohen Kappa coefficient

    # ROC AUC score with OvR strategy
    chart_ROC_AUC, roc_auc_per_label, roc_auc_ovr= compute_ROC_AUC_OVR(all_targets, all_preds_dist, AFFECTNET_CAT_EMOT)

    # Log the confusion matrix
    conf_matrix = confusion_matrix(all_targets.numpy(), all_preds_labels.numpy(), normalize = 'true')
    chart_conf_matrix = vis.create_conf_matrix(conf_matrix)

    # Log the metrics
    run.log({"Val accuracy per epoch": acc1,
             "Val top-2 accuracy per epoch": acc2,
             "Val global mean loss per epoch": global_epoch_loss,
             "Val F1-Score per epoch": f1_score,
             "Val Cohen Kappa coefficient per epoch": cohen_kappa,
             "Plot ROC AUC score with OvR strategy":  wandb.Image(chart_ROC_AUC),
             "Area Under the (ROC AUC) Curve per label": roc_auc_per_label,
             "Area Under the (ROC AUC) Curve OvR": roc_auc_ovr,
             "Confusion Matrix": chart_conf_matrix
             }, step=epoch+1, commit=True)

    metrics ={
        "Accuracy": acc1,
        "Top-2 Accuracy": acc2,
        "Global Val Mean Loss": global_epoch_loss,
        "F1-Score": f1_score,
        "Cohen Kappa coefficient": cohen_kappa,
        "Area Under the (ROC AUC) Curve OvR": roc_auc_ovr}
    return metrics



def save_checkpoint(
        state: 'dict', 
        is_best: bool,
        path: str = MODELS_DIR 
        ) -> None:
    torch.save(state, os.path.join(path,'checkpoint.pt'))
    
    # Save an extra copy if it is the best model yet
    if is_best:
        shutil.copyfile(os.path.join(path,'checkpoint.pt'), os.path.join(path,'model_best.pt'))  



def model_training(params = None):
    run = wandb.init(
    entity='armand-07',
    project="TFG Facial Emotion Recognition",
    job_type="train",
    config=params
    )
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    params = run.config
    run_name = run.name
    print(f'WanDB run name is: {run_name}') # Print the run number hash id

    # Set and clear the output directory
    saving_path = os.path.join(MODELS_DIR, f'run_{run_name}')
    if os.path.exists(saving_path):
        shutil.rmtree(saving_path)
    os.makedirs(saving_path)

    # Start measuring the emissions tracker
    tracker = EmissionsTracker(measure_power_secs=10, output_dir=saving_path,
                            log_level= "warning") # measure power every 10 seconds
    tracker.start()

    # Load the data
    print('Loading data...')
    dataloader_train, dataloader_val = data_loading(params)
    # Create and prepare the model and the optimizer
    print('Creating model and setting optimizer and criterion...')
    model, criterion, optimizer, device = model_creation(params)


    # Define the training parameters
    minimum_val_loss = 1000000.0 # Initialize the minimum validation loss with a high value
    best_f1_score = 0.0
    best_epoch = 0
    best_metrics = {}
    t0 = time.time()

    for epoch in range(params['epochs']):
        train(dataloader_train, model, criterion, optimizer, device, epoch, params, run)
        metrics = validate(dataloader_val, model, criterion, device, epoch, params, run)

        is_best = False # Flag to save the best model
        # Remember best f1-score and save checkpoint
        if metrics["Global Val Mean Loss"] < minimum_val_loss:
            is_best = True
            minimum_val_loss = metrics["Global Val Mean Loss"]
            best_f1_score = metrics["F1-Score"]
            best_epoch = epoch
            best_metrics = metrics

        save_checkpoint({
            'epoch': epoch+1,
            'arch': params['arch'],
            'state_dict': model.state_dict(),
            'f1_score': metrics["F1-Score"],
            'metrics': metrics,
            'params': dict(params),
        }, is_best, saving_path)
            
        # Training with early stopping if patience threshold is archieved
        if epoch - best_epoch == params['patience']:
            print(f'Early stopping at epoch {epoch+1}')
            t1 = time.time()
            print(f'Training time: {t1-t0:.2f} seconds')
            break

    # Stop measuring the emissions and show best results
    emissions: float = tracker.stop()
    run.log({'Emissions': emissions})
    print(f'Carbon emissions: {emissions} kg of CO2')
    print(f'Best F1-Score: {best_f1_score:.4f} at epoch {best_epoch+1}')
    print(best_metrics)

    # Save the best model to Weights and Biases
    artifact = wandb.Artifact(name=f"model_{run_name}", type="model", metadata=best_metrics)
    artifact.add_file(os.path.join(saving_path, 'model_best.pt'))
    artifact.add_file(os.path.join(saving_path, 'emissions.csv'))
    run.log_artifact(artifact)
    run.finish()



def main(mode, sweep_id):   
    wandb.login(key=wandbAPIkey)

    if mode == 'default':
        # Path of the parameters file
        params_path = Path("params.yaml")
        # Read data preparation parameters
        with open(params_path, "r", encoding='utf-8') as params_file:
            try:
                params = yaml.safe_load(params_file)
                params = params["training"]
            except yaml.YAMLError as exc:
                print(exc)
        model_training(params)

    elif mode == 'sweep':
        # Path of the parameters file
        config_sweep_path = Path("config_resnet50.yaml")
        # Read data preparation parameters
        with open(config_sweep_path, "r", encoding='utf-8') as config_file:
            try:
                sweep_configuration = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                print(exc)
        # 3: Start the sweep if it is not created
        if sweep_id is None:
            sweep_id = wandb.sweep(sweep=sweep_configuration, entity="armand-07", project="TFG Facial Emotion Recognition")
            print("Sweep id: ", sweep_id)
        else:
            sweep_id = "armand-07"+"/"+"TFG Facial Emotion Recognition"+"/"+sweep_id
        # 4: Run the sweep agent
        wandb.agent(sweep_id, model_training, count = 1)
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='default', help='The training mode to be sweep or standard run')
    parser.add_argument('--sweep_id', type=str, default=None, help='The sweep id to be used')
    return parser.parse_args() 



if __name__ == '__main__':
    args = parse_args()
    main(args.mode, args.sweep_id)