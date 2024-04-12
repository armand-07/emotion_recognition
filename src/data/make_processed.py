from pathlib import Path
import shutil
import yaml
import os

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import numpy as np
import cv2
import torch

from src import NUMBER_OF_EMOT, INTERIM_AFFECTNET_DIR, PROCESSED_AFFECTNET_DIR
from src.data.encoding import cart2polar_encoding
import src.data.compute_AffectNet_norm_values as compute_normalization_values


def generate_data(data_annot:pd.DataFrame, store_imgs:np.memmap, store_cat_emot:np.memmap, 
                  store_cont_emot:np.memmap, params:dict) -> None:
    """Generates the processed data and stores it in the memmap files. The ids are implicit in the order of the numpy array, 
    and this id is not equal to the original id. All the images are read and stored as a numpy array in the format [0-255]. 
    The categorical emotions are stored as a numpy array of int64. The continuous emotions are stored as a numpy array
    of float32. 

    Params:
        - data_annot (pd.DataFrame): The interim datasplit annotations ordered increasingly based on the path id of images. 
        - store_imgs (np.memmap): The memmap file to store the images
        - store_cat_emot (np.memmap): The memmap file to store the categorical emotions
        - store_cont_emot (np.memmap): The memmap file to store the continuous emotions
        - params (dict): The parameters to be used for the data processing
    Returns:
        - None
    """
    for id in tqdm(range(len(data_annot))):
        sample = data_annot.loc[id]                  # The id of the image is implicit
        img = cv2.imread(sample['path'])             # In BGR format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Convert from BGR to RGB
        store_imgs[id,:,:,:] = img[:,:,:]
        
        # Store the categorical emotion
        store_cat_emot[id] = np.int64(sample['cat_emot'])

        # Store the continuous emotion
        cont_emot = np.array([sample['valence'], sample['arousal']], dtype=np.float32)
        if params['continuous_format'] == 'cartesian': # No change is needed
            pass
        elif params['continuous_format'] == 'polar':
            cont_emot = cart2polar_encoding(cont_emot)

        store_cont_emot[id,:] = cont_emot[:]

    # Flush the data to the disk in order to store it 
    store_imgs.flush()
    store_cat_emot.flush()
    store_cont_emot.flush()


def generate_weights(data_annot:dict, output_path:str, datasplit:str) -> None:
    """Generates the weights of the categorical emotions and stores them in a tensor. It contains the 
    weights for each id in the datasplit based on the categorical emotions appearance in datasplit. 
    The weight is 1/count(cat_emot). 

    Params:
        - data_annot (pd.DataFrame): The interim datasplit annotations
        - output_path (str): The path where the processed datasplit will be stored
        - datasplit (str): The name of the datasplit
    Returns:
        - None
    """
    annotation_weights = data_annot['cat_emot'].value_counts().reset_index(name='count')
    annotation_weights['weight'] = 1 / annotation_weights['count']
    annotation_weights = annotation_weights.drop('count', axis=1).set_index('cat_emot')

    # Save the weights per label in a tensor
    label_weights = torch.zeros(NUMBER_OF_EMOT, dtype=torch.float32)
    for cat_emot in range(8):
        label_weights[cat_emot] = annotation_weights.loc[cat_emot]['weight'] # Assign the weight of the current label to the tensor
    torch.save(label_weights, os.path.join(output_path,'label_weights_' + datasplit + '.pt'))

    # Save the weights per sample in a tensor
    data_weights = torch.empty(len(data_annot))
    for idx in range(len(data_annot)):
        sample = data_annot.loc[idx]
        cat_emot = sample['cat_emot']
        weight = annotation_weights.loc[cat_emot]['weight']
        data_weights[idx] = torch.tensor(weight, dtype=torch.float64) # Assign the weight of the current sample to the tensor

    torch.save(data_weights, os.path.join(output_path,'data_weights_' + datasplit + '.pt'))


def process_datasplit(data_annot: pd.DataFrame, output_path:str, datasplit:str, params:dict) -> None:
    """From the interim datasplit annotations and the raw images, it generates the processed datasplit following 
    the 'params' specifications and stores it in a numpy memmap file. Concretly generates the following files:
    - datasplit_ids.dat: The ids of the images with the shape (N) and dtype int64
    - datasplit_imgs.dat: The images as numpy arrays in the shape (N, 224, 224, 3) and dtype uint8
    - datasplit_cat_emot.dat: The categorical emotions with the shape (N) and dtype int64
    - datasplit_cont_emot.dat: The continuous emotions with the shape (N, 2) and dtype float32
    It also generates the weights of the categorical emotions and stores them in a tensor. It is a (N) tensor 
    that contains the weights for each id in the datasplit.

    Parameters:
        - data_annot (pd.DataFrame): The interim datasplit annotations
        - output_path (str): The path where the processed datasplit will be stored
        - datasplit (str): The name of the datasplit
        - params (dict): The parameters to be used for the data processing
    Returns:
        - None
    """
    # Create the memmap files
    store_imgs = np.memmap(os.path.join(output_path, datasplit+"_imgs.dat"), dtype=np.uint8, 
                           mode='w+', shape=(len(data_annot), 224, 224, 3))
    store_cat_emot = np.memmap(os.path.join(output_path, datasplit+"_cat_emot.dat"), dtype=np.int64, 
                               mode='w+', shape=(len(data_annot)))
    store_cont_emot = np.memmap(os.path.join(output_path, datasplit+"_cont_emot.dat"), dtype=np.float32, 
                               mode='w+', shape=(len(data_annot), 2))
    # Order data annotations to try to read images sequentially:
    data_annot['sort_column'] = data_annot['path'].str.extract(r'(\d+)\.jpg$').astype(int)
    data_annot = data_annot.sort_values('sort_column').reset_index(drop=True)
    data_annot = data_annot.drop('sort_column', axis=1)

    # Generate the data
    generate_data(data_annot, store_imgs, store_cat_emot, store_cont_emot, params)
    # Generate the weights
    generate_weights(data_annot, output_path, datasplit)


def preprocess_affectnet(output_path:str, params:dict) -> None:
    """Preprocesses the affectnet dataset and its corresponding datasplits and stores the processed results. 
    It generates the processed datasplits following the 'params' specifications, the interim annotations 
    and the raw images and stores the results in numpy memmap files. The original training split is used as 
    the train/val set, the validation datasplit is used as the test. 
    It also computes the normalization values of the images (using val/train splits) and data weights and stores 
    them in .pt files. Morover, it stores the datasplit sizes in a CSV file.
    
    Params:
        - output_path (str): The path where the processed datasplit will be stored
        - params (dict): The parameters to be used for the data processing
    Returns:
        - None
    """
    # First load the interim annotations
    annotations_path = Path(os.path.join(INTERIM_AFFECTNET_DIR, 'annotations'))
    
    # Process the original training data to generate train/val splits
    print(f"Processing original training data to -> train/validation ({params['train_split']:.3f}/ {1 - params['train_split']:.3f})")
    file = os.path.join(annotations_path, 'train_set.pkl')
    data = pd.read_pickle(file)
    train_data, val_data = train_test_split(data, train_size=params['train_split'], random_state=params['random_seed'])
    train_data.reset_index(drop=True, inplace=True), val_data.reset_index(drop=True, inplace=True)
    process_datasplit(train_data, output_path, 'train', params)
    process_datasplit(val_data, output_path, 'val', params)

    # Process the original validation data to generate the test split
    print('Processing original validation data to -> test')
    file = os.path.join(annotations_path, 'val_set.pkl')
    test_data = pd.read_pickle(file)
    process_datasplit(test_data, output_path, 'test', params) # Save val split as test split

    # Store the datasplit sizes in a CSV file
    datasplit_sizes = {'train': len(train_data), 'val': len(val_data), 'test': len(test_data)}
    datasplit_sizes_df = pd.DataFrame(list(datasplit_sizes.items()), columns=['Datasplit', 'Size'])
    datasplit_sizes_df.to_csv(os.path.join (PROCESSED_AFFECTNET_DIR, 'datasplit_sizes.csv'), index=False)

    # Store the images normalization values in a .pt file
    compute_normalization_values.main(['train', 'val'], output_path)


def main(params:dict) -> None:
    """Runs data preprocessing script to turn interim data and raw images into
    cleaned data ready to be forwarded to model with a memory efficient data 
    structure. The output directory is cleaned before the new data processed 
    data is generated.
        
    Params:
        - params (dict): The parameters to be used for the data processing
    Returns:
        - None
    """
    output_path = PROCESSED_AFFECTNET_DIR
    # Delete the processed folder if it exists to clean creation of new data
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
           
    if 'affectnet' in params['orig_datasets']:
        preprocess_affectnet(output_path, params)
    else:
        raise ValueError("The dataset specified in the parameters is not supported for preprocessing.")


if __name__ == '__main__':
    # Path of the parameters file
    params_path = Path("params.yaml")
    # Read data preparation parameters
    with open(params_path, "r", encoding='utf-8') as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["preprocessing"]
        except yaml.YAMLError as exc:
            print(exc)
    main(params)