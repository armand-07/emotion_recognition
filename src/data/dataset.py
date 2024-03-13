from torch.utils.data import Dataset
from src import PROCESSED_COLUMNS
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.v2 as transforms
import cv2


class AffectNetDataset(Dataset):
    def __init__(self, annotations_path, img_transforms=None):
        self.annotations = pd.read_pickle(annotations_path)
        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations.iloc[idx]
        img = cv2.imread(data[PROCESSED_COLUMNS[1]])        # In BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # Convert from BGR to RGB
        if self.img_transforms is not None:
            img = self.img_transforms(image=img)["image"]

        cat_label = data[PROCESSED_COLUMNS[2]]
        cont_label = data[PROCESSED_COLUMNS[3]]

        return img, cat_label, cont_label            # Return the image and the continuous and categorical labels
    

class AffectNetDatasetValidation(Dataset):
    def __init__(self, annotations_path, img_transforms=None):
        self.annotations = pd.read_pickle(annotations_path)
        self.img_transforms = img_transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations.iloc[idx]
        id = data[PROCESSED_COLUMNS[0]]                     # Get the id of the image
        img = cv2.imread(data[PROCESSED_COLUMNS[1]])        # In BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # Convert from BGR to RGB
        if self.img_transforms is not None:
            img = self.img_transforms(image=img)["image"]

        cat_label = data[PROCESSED_COLUMNS[2]]
        cont_label = data[PROCESSED_COLUMNS[3]]

        return id, img, cat_label, cont_label        # Return the image and the continuous and categorical labels