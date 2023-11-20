# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import scipy.io
import os
import shutil

from src import RAW_DATA_DIR, INTERIM_DATA_DIR, INTERIM_COLUMNS



def process_interim_annotations(data, datasplit):
    """ Process annotations from mat file format to pandas DataFrame for given data split
    """
    data_annotations = [] # List of dictionaries with the annotations
    for id_key in range(len(data)):
        data_sample = data[id_key] # Get sample of the dataset

        # Get the image's filename, size and original database
        path = data_sample[1][0].split("/")[0]+"/"+data_sample[0][0] # data_sample[1][0] is the folder name and data_sample[0][0] is the filename, I delete images intermediate folder
        img_size = [int(data_sample[2][0][0][0][0][0]), int(data_sample[2][0][0][1][0][0])]
        orig_db = data_sample[3][0][0][0][0]

        # Get the image's people labelling, as there may be more than one person in the image
        label = data_sample[4][0] 
        people = len(label); 
        bbox = []; label_cat = []; label_cont = []; gender = []; age = [] # initialize list to store each people information
        for person in range(people):
            bbox.append([int(label[person][0][0][0]), int(label[person][0][0][1]), int(label[person][0][0][2]), int(label[person][0][0][3])])
            if datasplit == 'train':
                # Combined discrete labels
                label_person_cat = []
                if len(label[person][1]) > 0: # Check if there are labels
                    for i in range(len(label[person][1][0][0][0][0])):
                        label_person_cat.append(label[person][1][0][0][0][0][i][0])
                    label_cat.append(label_person_cat)
                else:
                    label_cat.append(['NA'])
            
                # Combined continious labels
                if not math.isnan(label[person][2][0][0][0][0][0]): # Check if there are labels (not NaN)
                    label_cont.append([int(label[person][2][0][0][0][0][0]),int(label[person][2][0][0][1][0][0]), int(label[person][2][0][0][2][0][0])])
                else:
                    label_cont.append(['NA', 'NA', 'NA'])

                # Gender and age
                gender.append(label[person][3][0][0])
                age.append(label[person][4][0][0])

            else: # Test and validation as they have different structure
                # Combined discrete labels
                label_person_cat = []
                if len(label[person][2]) > 0: # Check if there are labels
                    for i in range(len(label[person][2][0])):
                        label_person_cat.append(label[person][2][0][i][0])
                    label_cat.append(label_person_cat)
                else:
                    label_cat.append(['NA'])
            
                # Combined continious labels
                label_cont.append([int(label[person][4][0][0][0][0][0]),int(label[person][4][0][0][1][0][0]), int(label[person][4][0][0][2][0][0])])
                # Gender and age
                gender.append(label[person][5][0][0])
                age.append(label[person][6][0][0])

        # Create the pandas DataFrame with the collected information
        annotation_key = {'path': path, 'orig_db': orig_db, 'img_size': img_size,
        'people': people, 'bbox': bbox, 'label_cat': label_cat, 'label_cont': label_cont, 'gender': gender, 'age':age}
        data_annotations.append(annotation_key)
    return pd.DataFrame(data_annotations, columns = INTERIM_COLUMNS)


def copy_photos_to_interim(orig_data = RAW_DATA_DIR):
    """ Copy photos from raw to interim folder
    """
    source_folder = os.path.join(orig_data, 'PAMI', 'emotic', 'emotic')

    # Fetch all dataset directories
    for dataset_name in os.listdir(source_folder):
        # Construct full dataset directory path
        dataset_folder = os.path.join(source_folder, dataset_name, 'images')
        destination_folder = os.path.join(INTERIM_DATA_DIR, 'images', dataset_name)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Check if it is a directory
        if os.path.isdir(dataset_folder):
            for file_name in os.listdir(dataset_folder):
                # Copy only .jpg photos
                if os.path.isfile(os.path.join(dataset_folder, file_name)) and file_name.endswith('.jpg'):
                    source = os.path.join(dataset_folder, file_name)
                    destination = os.path.join(destination_folder, file_name)
                    shutil.copy(source, destination)


def main():
    """ Runs data processing scripts to turn raw data from (data/raw) into
        interim data ready to be preprocessed and make data explotation 
        (saved in data/interim).
    """
    # Delete the interim folder if it exists to clean creation of new data
    if os.path.exists(INTERIM_DATA_DIR):
        shutil.rmtree(INTERIM_DATA_DIR)

    os.makedirs(INTERIM_DATA_DIR)
    os.makedirs(os.path.join(INTERIM_DATA_DIR, "annotations"))



    # Load the dataset
    mat_path = os.path.join(RAW_DATA_DIR, 'PAMI', 'annotations', 'Annotations.mat')
    mat = scipy.io.loadmat(mat_path)
    print("---------- Generating interim dataset annotations ------------")
    print("Matlab information:", mat['__header__'])


    for datasplit in mat.keys():
        if datasplit in ['test', 'train', 'val']:
            print("Working on data split:", datasplit)
            dataframe_anotations = process_interim_annotations(mat[datasplit][0], datasplit)
            print("Total entries:", dataframe_anotations.shape[0])
            dataframe_anotations.to_pickle(os.path.join(INTERIM_DATA_DIR, "annotations", datasplit + '.pkl'))

    # Copy photos to interim folder
    print("---------- Copying photos to interim folder ------------")
    copy_photos_to_interim()
    print("---------- Finished generating interim dataset annotations ------------")
if __name__ == '__main__':
    main()
