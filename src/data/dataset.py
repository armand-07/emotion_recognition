from torch.utils.data import Dataset
from src import PROCESSED_COLUMNS
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import v2
import cv2


class AffectNetDataset(Dataset):
    def __init__(self, annotations_path, transform=None):
        self.annotations = pd.read_pickle(annotations_path)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations.iloc[idx]
        img_bgr = cv2.imread(data[PROCESSED_COLUMNS[1]])    # In BGR format
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)      # In RGB format
        img = img / 255.0                                   # Convert the image from 0-255 range to 0-1 range
        img = img.astype(np.float32)                        # Convert the image to float32
        img = img.transpose(2, 0, 1)                        # Convert image from [H,W,C] to [C,H,W] 

        if self.transform:
            image = self.transform(image)

        cat_label = data[PROCESSED_COLUMNS[2]]
        cont_label = data[PROCESSED_COLUMNS[3]]

        return img, cat_label, cont_label                   # Return the image and the continuous and categorical labels
    

class AffectNetDatasetValidation(Dataset):
    def __init__(self, annotations_path, transform=None):
        self.annotations = pd.read_pickle(annotations_path)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations.iloc[idx]
        id = data[PROCESSED_COLUMNS[0]]                                     # Get the id of the image
        img = cv2.imread(data[PROCESSED_COLUMNS[1]], cv2.COLOR_BGR2RGB)     # In RGB format
        tensor_img = v2.ToImage()(img)                                      # Convert numpy to a tensor, from [H,W,C] to [C,H,W] format

        if self.transform:
            tensor_img = self.transform(tensor_img)                         # Apply transformations to the image

        cat_label = data[PROCESSED_COLUMNS[2]]
        cont_label = data[PROCESSED_COLUMNS[3]]

        return id, tensor_img, cat_label, cont_label               # Return the image and the continuous and categorical labels