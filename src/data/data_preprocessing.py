from pathlib import Path
import yaml
import os
import pandas as pd


import torch
import numpy as np
import cv2
import torchvision.transforms as transforms

from src import INTERIM_DATA_DIR, INTERIM_COLUMNS, PROCESSED_COLUMNS
from src.models.face_detection_model import detect_faces_haar_cascade
from src.visualization.display_img_dataset import display_img_annot

def get_person_data_emotic(img, sample, people_idx, faces_bbox, params):
    """ Returns the person data for a given row of the emotic dataset following the 
    standard format of the processed dataset. 
    """
    body_bbox = sample['bbox'][people_idx]
    for f_bbox in faces_bbox:
        upper_left_point_inside = f_bbox[0] >= body_bbox[0] and f_bbox[1] >= body_bbox[1]
        down_right_point_inside = f_bbox[0] + f_bbox[2] <= body_bbox[2] and f_bbox[1] + f_bbox[3] <= body_bbox[3]
        if upper_left_point_inside and down_right_point_inside:
            # Crop the face from the image
            face_photo = img[f_bbox[1]:f_bbox[1]+f_bbox[3], f_bbox[0]:f_bbox[0]+f_bbox[2]]
            person_data = [face_photo, sample['label_cat'][people_idx], sample['label_cont'][people_idx]]
            arr = np.delete(faces_bbox, f_bbox) # Remove the face bbox from the list of faces
            return person_data, faces_bbox

    # If no face was detected return None
    return 'NA', faces_bbox




def data_preprocessing_emotic(params, pretrained_model):
    """ Preprocesses the emotic interim dataset and returns the processed rows.
    """
    processed_data = {column: [] for column in PROCESSED_COLUMNS}

    # Read the interim data annotations
    annotations_path = Path(os.path.join(INTERIM_DATA_DIR, 'annotations'))
    photo_directory = os.path.join(INTERIM_DATA_DIR, "images")
    annotations = {}
    for data_split in os.listdir(annotations_path):
        if data_split.endswith('.pkl'):
            file = os.path.join(annotations_path, data_split)
            data_part_name = data_split.split('.')[0]
            annotations[data_part_name] = pd.read_pickle(file)
    
    # Preprocess the data
    for data_split in annotations.keys():
        for photo_idx in range(len(annotations[data_split])):
            sample = annotations[data_split].loc[photo_idx]

            # Read the image
            img_file = sample['path']
            img_path = os.path.join(photo_directory, img_file)
            img = cv2.imread(img_path)

            display_img_annot(sample, bbox_thickness = 5, font_size = 2)
            # Detect faces in the image
            faces_bbox = detect_faces_haar_cascade(img, pretrained_model)
            print(faces_bbox)

            # Get the person data
            for person_idx in range(sample['people']):
                person_data, faces_bbox = get_person_data_emotic(img, sample, person_idx, faces_bbox, params)
                print(person_data)
                if person_data != 'NA':
                    processed_data['face_photo_tensor'].append(person_data[0])
                    processed_data['label_cat'].append(person_data[1])
                    processed_data['label_cont'].append(person_data[2])
                else:
                    print("No face was detected in the image: ", img_file)
                


def main(params):
    """ Runs data preprocessing scripts to turn interim data from (../data/interim) into
        cleaned data ready to be forwarded to model (saved in ../data/processed).
    """
    if params['face_detection_algorithm'] == "Haar Cascade Classifier":
        # Construct the path to the Haar cascade file
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        # Check if the file exists
        if os.path.exists(cascade_path):
            print("The Haar cascade file exists.")
        else:
            print("The Haar cascade file does not exist.")

        # Load the pre-trained face detection model
        pretrained_model = cv2.CascadeClassifier(cascade_path)
    else:
        assert False, "The face detection model is not valid."        

    if 'emotic' in params['orig_datasets']:
        data_preprocessing_emotic(params, pretrained_model)

    if 'affectnet' in params['processed_datasets']:
            pass
    

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