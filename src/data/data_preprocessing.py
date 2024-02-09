from pathlib import Path
import shutil
import yaml
import os

import pandas as pd
import datasets
from sklearn.model_selection import train_test_split

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from src import (INTERIM_DATA_DIR, INTERIM_COLUMNS_AFFECTNET, INTERIM_AFFECTNET_DIR, 
AFFECTNET_CAT_EMOT, PROCESSED_AFFECTNET_DIR, PROCESSED_COLUMNS)


def soft_label_encoding(y, classes, epsilon=0.1):
    y_soft = y * (1 - epsilon) + epsilon / classes
    return y_soft


def data_generator(data, params):
    """ Generator function for the data.
    """
    number_of_classes = len(AFFECTNET_CAT_EMOT)

    for idx in range(len(data)):
        sample = data.loc[idx]
        img_path = sample['path']
        id = img_path.split('/')[-1].split('.')[0]
            
        # Get the encoding of the categorical emotion
        cat_emot = F.one_hot(torch.tensor(sample['cat_emot']), num_classes=number_of_classes)
        if params['categorical_format'] == 'hard_label':
            pass # Do nothing
        elif params['categorical_format'] == 'soft_label':
            cat_emot = soft_label_encoding(cat_emot, number_of_classes)
        else:
            assert False, "The categorical format is not valid."

        # Get the encoding of the continuous emotions
        cont_emot = torch.tensor([sample['valence'], sample['arousal']])
        if params['continuous_format'] == 'cartesian':
            pass
        elif params['continuous_format'] == 'polar':
            # Convert to polar coordinates
            radius = torch.sqrt(cartesian_coords[:, 0]**2 + cartesian_coords[:, 1]**2)
            angle = torch.atan2(cartesian_coords[:, 1], cartesian_coords[:, 0])
            # Combine radius and angle into a single tensor
            cont_emot = torch.stack((radius, angle), dim=1)

        # Yield the result
        yield {PROCESSED_COLUMNS[0]: id, PROCESSED_COLUMNS[1]: img_path, 
               PROCESSED_COLUMNS[2]: cat_emot, PROCESSED_COLUMNS[3]: cont_emot}


def store_data_split(data, datasplit, params):
    """ Returns the processed data in Dataset format.
    """
    # Use the generator function
    annotations = pd.DataFrame(data_generator(data, params), columns=PROCESSED_COLUMNS)
    annotations.to_pickle(os.path.join(PROCESSED_AFFECTNET_DIR, datasplit + '.pkl'))


def data_preprocessing_affectnet(params):
    """ Preprocesses the affectnet interim dataset and stores the processed dataset in a pickle file.
    """
    annotations_path = Path(os.path.join(INTERIM_AFFECTNET_DIR, 'annotations'))

    # Process the training data
    print("------------- Processing training data ----------------")
    file = os.path.join(annotations_path, 'train_set.pkl')
    data = pd.read_pickle(file)
    # Split the DataFrame into training and validation sets
    train_data, val_data = train_test_split(data, train_size=params['train_split'], random_state=params['random_seed'])
    train_data.reset_index(drop=True, inplace=True), val_data.reset_index(drop=True, inplace=True)
    store_data_split(train_data,'train', params)
    store_data_split(val_data, 'val', params)

    # Process the validation data
    print("------------- Processing validation data ----------------")
    file = os.path.join(annotations_path, 'val_set.pkl')
    test_data = pd.read_pickle(file)
    store_data_split(test_data, 'test', params) # Save val split as test split


def main(params):
    """ Runs data preprocessing scripts to turn interim data from (../data/interim) into
        cleaned data ready to be forwarded to model (saved in ../data/processed).
    """
    # Delete the processed folder if it exists to clean creation of new data
    if os.path.exists(PROCESSED_AFFECTNET_DIR):
        shutil.rmtree(PROCESSED_AFFECTNET_DIR)
    # Ensure all the required directories exist
    if not os.path.exists(PROCESSED_AFFECTNET_DIR):
        os.mkdir(PROCESSED_AFFECTNET_DIR)

    if params['face_detection_algorithm'] == "Haar Cascade Classifier":
        # Construct the path to the Haar cascade file
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        # Check if the file exists
        if os.path.exists(cascade_path):
            print("The Haar cascade model file exists.")
        else:
            print("The Haar cascade model file does not exist.")

        # Load the pre-trained face detection model
        pretrained_model = cv2.CascadeClassifier(cascade_path)
    elif params['face_detection_algorithm'] == "None":
        pretrained_model = None
        print("No face detection model will be used.")
    else:
        assert False, "The face detection model is not valid."        

    if 'emotic' in params['orig_datasets']:
        pass

    if 'affectnet' in params['orig_datasets']:
        data_preprocessing_affectnet(params)
    

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