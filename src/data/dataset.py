from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
import albumentations
import albumentations as A
from torch.utils.data import DataLoader, WeightedRandomSampler
from albumentations.pytorch import ToTensorV2
import cv2

from src import PROCESSED_AFFECTNET_DIR, DATA_DIR


class AffectNetDataset(Dataset):
    def __init__(self, path:str, datasplit:str,  img_transforms:albumentations.Compose=None):
        # Get datasplit size
        datasplit_sizes_df = pd.read_csv(os.path.join(path,"datasplit_sizes.csv"))
        self.datasplit_len = datasplit_sizes_df[datasplit_sizes_df['Datasplit'] == datasplit]['Size'].values[0]

        # Load the data
        datasplit_path = os.path.join(path, datasplit)
        self.store_imgs = np.memmap(datasplit_path+"_imgs.dat", dtype=np.uint8, 
                            mode='r', shape=(self.datasplit_len, 224, 224, 3))
        self.store_cat_emot = np.memmap(datasplit_path+"_cat_emot.dat", dtype=np.int64, 
                                mode='r+', shape=(self.datasplit_len,1))
        self.store_cont_emot = np.memmap(datasplit_path+"_cont_emot.dat", dtype=np.float32, 
                                mode='r+', shape=(self.datasplit_len, 2))
        self.img_transforms = img_transforms

    def __len__(self):
        return self.datasplit_len
    
    def __getitem__(self, idx:int):
        img = self.store_imgs[idx]
        if self.img_transforms is not None:
            img = self.img_transforms(image=img)["image"]
        
        cat_label = torch.from_numpy(self.store_cat_emot[idx])[0]
        cont_label = torch.from_numpy(self.store_cont_emot[idx])

        return img, cat_label, cont_label            # Return the image and the continuous and categorical labels
    


class AffectNetDatasetValidation(Dataset):
    """AffectNet dataset for validation and test. The dataset is loaded in memory and the images are not transformed until 
    "getitem" is called."""
    def __init__(self, path:str, datasplit:str, img_transforms:albumentations.Compose = None):
        # Get datasplit size
        datasplit_sizes_df = pd.read_csv(os.path.join(path,"datasplit_sizes.csv"))
        self.datasplit_len = datasplit_sizes_df[datasplit_sizes_df['Datasplit'] == datasplit]['Size'].values[0]

        # Load the data
        datasplit_path = os.path.join(path, datasplit)
        self.store_imgs = np.memmap(datasplit_path+"_imgs.dat", dtype=np.uint8, 
                            mode='r', shape=(self.datasplit_len, 224, 224, 3))
        self.store_cat_emot = np.memmap(datasplit_path+"_cat_emot.dat", dtype=np.int64, 
                                mode='r+', shape=(self.datasplit_len,1))
        self.store_cont_emot = np.memmap(datasplit_path+"_cont_emot.dat", dtype=np.float32, 
                                mode='r+', shape=(self.datasplit_len, 2))
        self.img_transforms = img_transforms

    def __len__(self):
        return self.datasplit_len
    
    def __getitem__(self, idx:int):
        img = self.store_imgs[idx]
        if self.img_transforms is not None:
            img = self.img_transforms(image=img)["image"]
        
        cat_label = torch.from_numpy(self.store_cat_emot[idx])[0]
        cont_label = torch.from_numpy(self.store_cont_emot[idx])

        return img, cat_label, cont_label, idx
    


def data_transforms(only_normalize:bool = False, daug_params:dict = dict(), image_norm:str = "imagenet", resize:bool = False) -> albumentations.Compose:
    """Create the data transformations for the images. The transformations are applied in the following order:
        1. Horizontal flip
        2. Shift, scale and rotate
        3. Coarse dropout
        4. Color jitter
        5. Gaussian noise
        6. Resize
        7. Normalize
        8. Convert to Pytorch tensor
    Parameters:
        - only_normalize: bool. If True, only the normalization is applied to the image
        - daug_params: dict. Dictionary with the data augmentation parameters
        - image_norm: str. The normalization method to apply to the image. Can be "imagenet", "affectnet" or "none"
        - resize: bool. If True, the image is resized to 224x224
    Returns:
        - A.Compose object with the transformations to apply to the image
    """
    transforms = []
    if not only_normalize:
        p_value = daug_params["daug_p_value"]
        if daug_params["daug_horizontalflip"]:
            transforms.append(A.HorizontalFlip(p = p_value))
        if daug_params["daug_shiftscalerotate"]:
            transforms.append(A.ShiftScaleRotate(rotate_limit=(-15, 15), 
                shift_limit=(0, 0.1), scale_limit=(-0.1, 0.1), 
                border_mode = cv2.BORDER_CONSTANT, value = 0.0, p = p_value))
        if daug_params["daug_coarsedropout"]:
            transforms.append(A.CoarseDropout(max_height=85, min_height = 16, 
                max_width = 85, min_width = 16, fill_value = 0.0, max_holes = 1, 
                min_holes = 1, p = p_value))
        if daug_params["daug_colorjitter"]:
            transforms.append(A.ColorJitter(brightness = [0.85, 1.15], 
                contrast = [0.9,1.1], saturation = [0.75,1.1], hue = [-0.01,0.02], 
                p = p_value))
        if daug_params["daug_gaussnoise"]:
            transforms.append(A.GaussNoise(var_limit = (10.0, 75.0), p = p_value))
    if resize:
        transforms.append(A.Resize(224, 224))

    # Normalize the image
    if image_norm.lower() == "imagenet":      # Normalize the image with the mean and std of the ImageNet dataset
        transforms.append(A.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]))
    elif image_norm.lower() == "affectnet":   # Normalize the image with the mean and std of the AffectNet dataset
        norm_values_path = os.path.join (PROCESSED_AFFECTNET_DIR, 'dataset_normalization_values.pt')
        if os.path.exists(norm_values_path):
            normalization_values = torch.load(norm_values_path)
        else: 
            norm_values_path = os.path.join (DATA_DIR, 'affectnet', 'dataset_normalization_values.pt')
            if os.path.exists(norm_values_path):
                normalization_values = torch.load(norm_values_path)
            else:
                raise ValueError("Normalization values for AffectNet dataset not found")
            
            
        transforms.append(A.Normalize(mean=normalization_values['mean'], 
            std=normalization_values['std']))
    elif image_norm.lower() == "none":        # Do not normalize the image, only to [0-1] range
        transforms.append(A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]))
    else:
        raise ValueError(f"Invalid image_norm parameter: {image_norm}")
    # Add to list the conversion to Pytorch tensor and convert the list of transforms to a Compose object
    transforms.append(ToTensorV2())
    return A.Compose(transforms) 



def create_dataloader(datasplit, batch_size, weighted_dataloader = False, epoch_samples = "original", 
                      daug_params = dict(), image_norm = "imagenet", num_workers = 4):
    """Create a dataloader for the AffectNet dataset with the specified datasplit and batch size. The dataloader can be
    weighted or not, depending on the weighted_dataloader parameter. The data augmentation parameters can be specified
    with the daug_params parameter that is a dictionary. The image normalization can be specified with the image_norm parameter.
    Parameters:
        - datasplit: str. The datasplit to use. Can be "train", "val" or "test"
        - batch_size: int. The batch size to use
        - weighted_dataloader: bool. If True, a weighted dataloader is created
        - epoch_samples: int or str. The number of samples to use in an epoch. If "original", the original number of samples
            in the datasplit is used
        - daug_params: dict. Dictionary with the data augmentation parameters
        - image_norm: str. The normalization method to apply to the image. Can be "imagenet", "affectnet" or "none"
        - num_workers: int. The number of workers to use in the dataloader. The recommended value is 4. 
    Returns:
        - DataLoader object with the AffectNet dataset
    """
    # Only apply data augmentation on test and val if the weighted dataloader is used 
    if not weighted_dataloader and (datasplit == "test" or datasplit == "val"):
        transforms = data_transforms(only_normalize = True, daug_params = daug_params, image_norm = image_norm)
    else:
        transforms = data_transforms(daug_params = daug_params, image_norm = image_norm)
    # Create the datasets
    dataset = AffectNetDataset(path=PROCESSED_AFFECTNET_DIR, datasplit = datasplit,
                                    img_transforms=transforms)
    if weighted_dataloader:
        if epoch_samples == "original":
            epoch_size = len(dataset)
        else:
            epoch_size = epoch_samples
        # Load weights
        weights = torch.load(os.path.join(PROCESSED_AFFECTNET_DIR, "data_weights_" + datasplit + ".pt"))
        # Load sampler
        sampler = WeightedRandomSampler(weights, epoch_size, replacement=True)
        # Create dataloaders
        dataloader = DataLoader(dataset, batch_size = batch_size, pin_memory = True,
                                sampler = sampler, drop_last = True, num_workers = num_workers)
        print(f"Weighted dataloader created for {datasplit} datasplit")
        print(f"Transforms: {transforms}")
    else:
        dataloader = DataLoader(dataset, batch_size = batch_size, pin_memory = True,
                                shuffle = True, drop_last = True, num_workers = num_workers)
        print(f"Normal dataloader created for {datasplit} datasplit")
        print(f"Transforms: {transforms}")
    return dataloader