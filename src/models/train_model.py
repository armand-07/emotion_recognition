from pathlib import Path
from tqdm import tqdm
import yaml
import os
import time
import shutil
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from torch.nn import functional as F


from codecarbon import EmissionsTracker
import wandb
from wandb.sdk import wandb_run

import math

from src import MODELS_DIR
from src.data.dataset import create_dataloader
from src.models import architectures as arch
from src.models.metrics import save_val_wandb_metrics, save_val_wandb_metrics_dist

from config import wandbAPIkey



def train(train_loader: DataLoader, model: torch.nn.Module, criterion: torch.nn, optimizer: torch.optim, 
        device: torch.device, epoch: int, params: dict, run: wandb_run.Run) -> None:
    """Function to train the model for one epoch. It computes the loss and the accuracy 
    of the model and logs them to Weights and Biases.
    Parameters:
        - train_loader: DataLoader with the training data
        - model: Pytorch model to be trained
        - criterion: Loss function to be used
        - optimizer: Pytorch optimizer to be used
        - device: Pytorch device where the model is located
        - epoch: Current epoch number
        - params: Dictionary with the parameters of the training
        - run: Weights and Biases run object
    Returns:
        - None"""
    # Switch to train mode
    model.train()
    # Stablish some metrics to be saved during training
    global_epoch_loss = torch.zeros(1, dtype=torch.float, device = device)
    acc = MulticlassAccuracy(device=device)
    
    for i, (imgs, cat_target, _) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                    desc = f'(TRAIN)Epoch {epoch+1}', 
                                                    miniters=int(len(train_loader)/100)):
        optimizer.zero_grad() # reset gradients
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
        if i % 100 == 0: # Print the metrics every 100 batches
            predicted_label = torch.argmax(prediction, dim=1)
            acc_batch = torch.sum(predicted_label == cat_target).item()/params['batch_size']*100
            tqdm.write(f"TRAIN [{i+1}/{len(train_loader)}], Batch accuracy: {acc_batch:.3f}%; Batch Loss: {loss.item():.3f}")
            if torch.isnan(loss): # Check if the loss is NaN to stop training
                raise ValueError(f"Loss is NaN at epoch {epoch+1} and step {i+1}, caused by gradient clipping or small learning rate")
            
    # Compute the metrics
    acc = acc.compute().item()
    global_epoch_loss = global_epoch_loss / (len(train_loader)) # all batches have same size
    run.log({"Train accuracy per epoch": acc}, step=epoch+1)
    run.log({"Train mean loss per epoch": global_epoch_loss}, step=epoch+1)



def train_distillation(train_loader: DataLoader, model_student: torch.nn.Module, model_teacher: torch.nn.Module, 
        label_criterion: torch.nn, distill_criterion: torch.nn, alpha: float,optimizer: torch.optim, 
        device: torch.device, epoch: int, params: dict, run: wandb_run.Run) -> None:
    """Function to train the model for one epoch using distillation. It computes the loss, cosine similarity between 
    the label and distillation output embedding and the accuracy of the model and logs them to Weights and Biases.
    Parameters:
        - train_loader: DataLoader with the training data
        - model_student: Pytorch model to be trained
        - model_teacher: Pytorch model to be used as teacher
        - label_criterion: Loss function to be used for the labels
        - distill_criterion: Loss function to be used for the distillation
        - alpha: Weight of the distillation loss
        - optimizer: Pytorch optimizer to be used
        - device: Pytorch device where the model is located
        - epoch: Current epoch number
        - params: Dictionary with the parameters of the training
        - run: Weights and Biases run object
    Returns:
        - None
    """
    # Set models to correct mode
    model_student.train()
    model_teacher.eval()
    
    # Stablish some metrics to be saved during training
    global_epoch_loss = torch.zeros(1, dtype=torch.float, device = device)
    acc = MulticlassAccuracy(device=device)
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    global_cosine_sim = torch.zeros(1, dtype=torch.float, device = device)
    
    for i, (imgs, cat_target, _) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                    desc = f'(TRAIN)Epoch {epoch+1}', 
                                                    miniters=int(len(train_loader)/100)):
        optimizer.zero_grad() # reset gradients
        # Move images and target to gpu
        imgs = imgs.to(device)
        cat_target = cat_target.to(device)

        # Forward batch of images through the network
        pred_student, pred_dist_student = model_student(imgs)
         
        with torch.no_grad():
            pred_teacher = model_teacher(imgs)
            pred_teacher = F.softmax(pred_teacher, dim=1) # Logits converted to probs            
        # Compute the loss
        loss_label = label_criterion(pred_student, cat_target)
        loss_distill = distill_criterion(pred_dist_student, pred_teacher)
        loss = (1-alpha) * loss_label + alpha * loss_distill
        
        # Compute gradient and optimize weights
        loss.backward()
        optimizer.step()

        # Measure accuracy and loss
        acc.update(pred_student, cat_target)
        global_epoch_loss += loss
        global_cosine_sim += torch.sum (cos_sim(pred_student, pred_dist_student)) 
        if i % 100 == 0: # Print the metrics every 100 batches
            predicted_label = torch.argmax(pred_student, dim=1)
            acc_batch = torch.sum(predicted_label == cat_target).item()/params['batch_size']*100
            tqdm.write(f"TRAIN [{i+1}/{len(train_loader)}], Batch accuracy: {acc_batch:.3f}%; Batch Loss: {loss.item():.3f}")
            if torch.isnan(loss): # Check if the loss is NaN to stop training
                raise ValueError(f"Loss is NaN at epoch {epoch+1} and step {i+1}, caused by gradient clipping or small learning rate")

    # Compute the metrics
    acc = acc.compute().item()
    global_epoch_loss = global_epoch_loss / (len(train_loader)) # all batches have same size
    global_cosine_sim = global_cosine_sim / (len(train_loader)*params['batch_size']) # all batches have same size
    run.log({"Train accuracy per epoch": acc}, step=epoch+1)
    run.log({"Train mean loss per epoch": global_epoch_loss}, step=epoch+1)
    run.log({"Train output embedding label/distill cosine similarity per epoch": global_cosine_sim}, step=epoch+1)



def validate(val_loader: DataLoader, model: torch.nn.Module, criterion: torch.nn, device: torch.device,
             epoch: int, batch_size: int, run: wandb_run.Run) -> dict:
    """
    Function to validate the model for one epoch. It computes many metrics and logs them to Weights and Biases and returns them.
    Parameters:
        - val_loader: DataLoader with the validation data
        - model: Pytorch model to be validated
        - criterion: Loss function to be used
        - device: Pytorch device where the model is located
        - epoch: Current epoch number
        - batch_size: The size of the batches
        - run: Weights and Biases run object
        Returns:
        - metrics: Dictionary with the metrics computed during the validation
    """
    model.eval() # switch model to evaluate mode
    # Stablish some metrics to be saved during validation
    acc1 = torch.zeros(1, dtype=torch.int, device = device)
    all_preds_labels = torch.empty(0, device = 'cpu')
    all_preds_distrib = torch.empty(0, device = 'cpu')
    softmax = nn.Softmax(dim=1)
    all_targets = torch.empty(0, device = 'cpu')
    global_epoch_loss = 0.0
    
    with torch.no_grad(): # No need to compute gradients
        for i, (imgs, cat_target, _) in tqdm(enumerate(val_loader), total=len(val_loader),
                                                        desc = f'(VAL)Epoch {epoch+1}', 
                                                        miniters=int(len(val_loader)/100)):
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

            # Store the predictions and targets to compute metrics
            all_preds_labels = torch.cat((all_preds_labels, predicted_label.cpu())) 
            all_preds_distrib = torch.cat((all_preds_distrib, softmax(prediction).cpu())) # Apply softmax to the predictions as they are in logits
            all_targets = torch.cat((all_targets, cat_target.cpu()))
            
            if i % 100 == 0: # Print the metrics every 100 batches
                acc_batch = torch.sum(predicted_label == cat_target).item()/batch_size*100
                tqdm.write(f'VAL [{i+1}/{len(val_loader)}], Batch accuracy: {acc_batch:.2f}%; Batch Loss: {loss.item():.3f}')
        # Compute metrics
        metrics = save_val_wandb_metrics(acc1, val_loader, batch_size, all_targets, all_preds_distrib,
                                 all_preds_labels, global_epoch_loss, epoch, run)
    return metrics



def validate_distillation(val_loader: DataLoader, model: torch.nn.Module, criterion: torch.nn, embedding_method: str,
                        device: torch.device, epoch: int, batch_size: int, run: wandb_run.Run) -> dict:
    """
    Function to validate the model for one epoch using distillation. It computes many metrics and logs them to Weights 
    and Biases and returns them.
    Parameters:
        - val_loader: DataLoader with the validation data
        - model: Pytorch model to be validated
        - criterion: Loss function to be used
        - embedding_method: The method to use to compute the embedding
        - device: Pytorch device where the model is located
        - epoch: Current epoch number
        - batch_size: The size of the batches
        - run: Weights and Biases run object
        Returns:
        - metrics: Dictionary with the metrics computed during the validation
    """
    model.eval() # switch model to evaluate mode
    # Stablish some metrics to be saved during validation
    acc1 = torch.zeros(1, dtype=torch.int, device = device)
    all_preds_labels = torch.empty(0, device = 'cpu')
    all_preds_dist = torch.empty(0, device = 'cpu')
    softmax = nn.Softmax(dim=1)
    all_targets = torch.empty(0, device = 'cpu')
    global_epoch_loss = 0.0
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    global_cosine_sim = torch.zeros(1, dtype=torch.float, device = device)
    
    with torch.no_grad(): # No need to compute gradients
        for i, (imgs, cat_target, _) in tqdm(enumerate(val_loader), 
                                                       total=len(val_loader), desc = f'(VAL)Epoch {epoch+1}', 
                                                        miniters=int(len(val_loader)/100)):
            # Move images to gpu
            imgs = imgs.to(device)
            cat_target = cat_target.to(device)

            # Forward batch of images through the network
            pred, pred_dist = model(imgs)
            if embedding_method == "class":
                prediction = pred
            elif embedding_method == "distill":
                prediction = pred_dist
            elif embedding_method == "both":
                prediction = pred + pred_dist

            predicted_label = torch.argmax(prediction, dim=1) # Get the predicted labels using argmax
            
            # Compute the loss
            loss = criterion(prediction, cat_target)
            # Measure metrics
            acc1 += torch.sum(predicted_label == cat_target)
            global_epoch_loss += loss.data.item()*imgs.shape[0] # Accumulate the loss
            global_cosine_sim += torch.sum(cos_sim (pred, pred_dist))

            # Store the predictions and targets to compute metrics
            all_preds_labels = torch.cat((all_preds_labels, predicted_label.cpu())) 
            all_preds_dist = torch.cat((all_preds_dist, softmax(prediction).cpu())) # Apply softmax to the predictions as they are in logits
            all_targets = torch.cat((all_targets, cat_target.cpu()))
            
            if i % 100 == 0: # Print the metrics every 100 batches
                acc_batch = torch.sum(predicted_label == cat_target).item()/batch_size*100
                tqdm.write(f'VAL [{i+1}/{len(val_loader)}], Batch accuracy: {acc_batch:.2f}%; Batch Loss: {loss.item():.3f}')
        
        # Compute metrics
        metrics = save_val_wandb_metrics_dist(acc1, val_loader, batch_size, all_targets, all_preds_dist, 
                           all_preds_labels, global_epoch_loss, epoch, global_cosine_sim,
                           run)
    return metrics



def save_checkpoint(state: 'dict', is_best: bool, path: str = MODELS_DIR) -> None:
    """
    Function to save the model checkpoint. It saves the model with it's state. 
    If the model is the best, it saves an extra copy.
    Parameters:
        - state: Dictionary with the model state, it should contain: model weights, epoch, metrics, and parameters. 
        - is_best: Flag to save the best model
        - path: Path where the model will be saved
    Returns:
        - None
    """
    torch.save(state, os.path.join(path,'checkpoint.pt'))
    
    # Save an extra copy if it is the best model yet
    if is_best:
        shutil.copyfile(os.path.join(path,'checkpoint.pt'), os.path.join(path,'model_best.pt'))  



def model_training(params = None):
    """
    Function to train a model using the parameters provided. It logs the results to Weights and Biases and saves the best model. 
    This function is able to perform a standard training (using params.yaml config), an extensive training from a wandb training 
    (taking its parameters and restart training), or a sweep training as a wandb agent.
    Parameters:
        - params: Dictionary with the parameters of the training. If none is provided, this config will be set by Sweep Controller 
                by wandb.agent
    Returns:
        - None
    """
    if params is None: # If no parameters are provided, set the job_type to train
        job_type = "train"
    elif 'job_type' in params: # If the job_type is provided, use it
        job_type = params['job_type'] 
        params.pop('job_type')
    run = wandb.init(
    entity='armand-07',
    project="TFG Facial Emotion Recognition",
    job_type=job_type,
    config=params
    )
    # If called by wandb.agent, this config will be set by Sweep Controller
    params = run.config
    run_name = run.name
    print(f'WanDB run name is: {run_name}')

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
    daug_params = {k: v for k, v in params.items() if k.startswith('daug')} # Get the data augmentation parameters

    if 'weighted_sampler_train' in params:
        weighted_data_train = params['weighted_sampler_train']
    elif 'weighted_train' in params:
        weighted_data_train = params['weighted_train']
    else:
        weighted_data_train = True

    if 'weighted_sampler_val' in params:
        weighted_data_val = params['weighted_sampler_val']
    elif 'weighted_val' in params:
        weighted_data_val = params['weighted_val']
    else:
        weighted_data_val = False


    dataloader_train = create_dataloader (datasplit = "train", batch_size = params['batch_size'], 
                                            weighted_dataloader = weighted_data_train, 
                                            epoch_samples = params['epoch_samples'], daug_params = daug_params, 
                                            image_norm = params['image_norm'], num_workers = 4)
    dataloader_val = create_dataloader (datasplit = "val", batch_size = params['batch_size'],
                                            weighted_dataloader = weighted_data_val,
                                            epoch_samples = params['epoch_samples'], daug_params = daug_params,
                                            image_norm = params['image_norm'], num_workers = 4)

    # Create and prepare the model and the optimizer
    print('Creating model and setting optimizer...')
    arch.seed_everything(params['random_seed'])
    model, device = arch.model_creation(params['arch'], weights = params['pretraining'])
    optimizer = arch.define_optimizer(model, params['optimizer'], params['lr'], params['momentum'])

    # Define the criterion for train and val
    print('Defining the criterion...')
    if 'label_smoothing' not in params:
        label_smoothing = 0.0
    else:
        label_smoothing = params['label_smoothing']
    criterion_train = arch.define_criterion(params, label_smoothing, 'train', device = device)
    criterion_val = arch.define_criterion(params, label_smoothing, 'val', device = device)

    # Define the distillation parameters
    if 'distillation' in params and params['distillation']:
        print('Distillation is enabled, loading teacher model and distillation criterion...')
        distillation = True
        model_teacher, _ = arch.model_creation(params['teacher_arch'], weights = 'affectnet_cat_emot', device = device)
        model_teacher.eval()
        criterion_distill = arch.define_criterion(params, params['label_smoothing_dist'], 'train', distillation = True, device = device)
        alpha = params['alpha']
    else:
        distillation = False

    # Define the training parameters
    minimum_val_loss = 1000000.0 # Initialize the minimum validation loss with a high value
    f1_score = 0.0
    best_epoch = 0
    best_metrics = {}
    t0 = time.time()

    for epoch in range(params['epochs']):
        if distillation:
            decaying_strategy = int(params['decaying_strategy'])
            if decaying_strategy == 0:
                alpha = params['alpha']
            elif decaying_strategy == 1:
                alpha = (1 - (epoch/params['epochs']))*params['alpha']
            elif decaying_strategy == 2:
                alpha = (math.cos(epoch/params['epochs']))*params['alpha']
            elif decaying_strategy == 3:
                alpha = ((1 - epoch/params['epochs'])**2)*params['alpha']
            else:
                raise ValueError(f"Invalid decaying strategy parameter: {decaying_strategy}")
            
            train_distillation(dataloader_train, model, model_teacher, criterion_train, criterion_distill, 
                               alpha, optimizer, device, epoch, params, run)
            metrics = validate_distillation(dataloader_val, model, criterion_val, params['embedding_method'], 
                                            device, epoch, params['batch_size'], run)
        else:
            train(dataloader_train, model, criterion_train, optimizer, device, epoch, params, run)
            metrics = validate(dataloader_val, model, criterion_val, device, epoch, params['batch_size'], run)

        is_best = False # Flag to save the best model
        # Remember best f1-score and save checkpoint
        if metrics["Global Val Mean Loss"] < minimum_val_loss:
            is_best = True
            minimum_val_loss = metrics["Global Val Mean Loss"]
            f1_score = metrics["F1-Score"]
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
    print(f'Mininum val loss: {minimum_val_loss:.4f} at epoch {best_epoch+1}, with F1-Score: {f1_score:.4f}')
    print(best_metrics)

    # Save the best model to Weights and Biases
    artifact = wandb.Artifact(name=f"model_{run_name}", type="model", metadata=best_metrics)
    artifact.add_file(os.path.join(saving_path, 'model_best.pt'))
    artifact.add_file(os.path.join(saving_path, 'emissions.csv'))
    run.log_artifact(artifact)
    run.finish()



def main(mode, wandb_id):   
    """
    Main function to preprare the training of the model. It can be used in three different modes: 
    default, sweep or extensive_train.
    Parameters:
        - mode: The training mode to be used
        - wandb_id: The sweep id to be used if the mode is sweep, or the run id to be used if the mode is extensive_train
    Returns:
        - None"""
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
        params['job_type'] = "train"
        model_training(params)
        
    elif mode == 'sweep':
        # Path of the parameters file
        config_sweep_path = Path("config_resnet50_pretrained_weighted_loss.yaml")
        # Read data preparation parameters
        with open(config_sweep_path, "r", encoding='utf-8') as config_file:
            try:
                sweep_configuration = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                print(exc)
        # 3: Start the sweep if it is not created
        if wandb_id is None:
            wandb_id = wandb.sweep(sweep=sweep_configuration, entity="armand-07", project="TFG Facial Emotion Recognition")
            print("Sweep id: ", wandb_id)
        else:
            wandb_id = "armand-07"+"/"+"TFG Facial Emotion Recognition"+"/"+wandb_id
        # 4: Run the sweep agent
        wandb.agent(wandb_id, model_training, count = 1)

    elif mode == "extensive_train":
        # Start api
        api = wandb.Api()
        # Get run params
        try: 
            if not wandb_id.startswith("armand-07/TFG Facial Emotion Recognition/"):
                full_wandb_id = "armand-07/TFG Facial Emotion Recognition/" + wandb_id
            run = api.run(full_wandb_id)
            params = run.config
        except:
            print("Run id not provided, trying to acces the run by artifact's name")
            if not wandb_id.startswith("armand-07/TFG Facial Emotion Recognition/model_"):
                full_wandb_id = "armand-07/TFG Facial Emotion Recognition/model_" + wandb_id + ":latest"
            artifact = api.artifact(full_wandb_id)
            artifact_dir = artifact.download()
            print(f'Artifact downloaded to: {artifact_dir}')
            # Path of the parameters file
            local_artifact = torch.load(os.path.join(artifact_dir, "model_best.pt"))
            params = local_artifact["params"]
        # Increase the number of epochs and patience
        params['epochs'] = 100
        params['patience'] = 10
        params['job_type'] = "extensive_train"
        print(params)
        model_training(params = params)
        
    else:
        raise ValueError(f"Invalid mode parameter: {mode}")



def parse_args():
    """Function to parse the arguments of the command line. It returns the arguments as a Namespace object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='default', help='The training mode to be sweep or standard run')
    parser.add_argument('--wandb_id', type=str, default=None, help='The sweep id to be used')
    return parser.parse_args() 



if __name__ == '__main__':
    args = parse_args()
    main(args.mode, args.wandb_id)