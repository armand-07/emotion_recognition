from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
import albumentations


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