# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import scipy.io
import os
import shutil

from src import RAW_DATA_DIR, RAW_AFFECTNET_DIR, INTERIM_DATA_DIR, INTERIM_AFFECTNET_DIR, INTERIM_COLUMNS_PAMI, INTERIM_COLUMNS_AFFECTNET



def pami_process_interim_annotations(datasplit, data):
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
                    label_cont.append([np.nan, np.nan, np.nan])

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
    return pd.DataFrame(data_annotations, columns = INTERIM_COLUMNS_PAMI)


def affectnet_process_interim_annotations(id_list, datasplit_path):
    """ Process annotations from mat file format to pandas DataFrame for validation and train"""

    data_annotations = [] # List of dictionaries with the annotations
    
    filenames = ['aro', 'lnd', 'val', 'exp']
    for id in id_list:
        exp = 0; valence = 0; arousal = 0
        for filename in filenames:
            data = np.load(os.path.join(datasplit_path, "annotations", id + "_" + filename + ".npy"))
            if filename == "exp":
                exp = int(data.item())
            elif filename == "valence":
                valence = float(data.item())
            elif filename == "arousal":
                arousal = float(data.item())
        id_image_path = os.path.join(datasplit_path, "images", id + ".jpg")
        annotation_key = {'path': id_image_path, 'label_cat': exp, 'val': valence, 'aro': arousal}
        data_annotations.append(annotation_key)
    return pd.DataFrame(data_annotations, columns = INTERIM_COLUMNS_AFFECTNET)


def copy_images_to_interim(source_folder, dest_folder):
    """ Copy images from raw to interim folder
    """

    # Fetch all dataset directories
    for datasplit_name in os.listdir(source_folder):
        # Construct full dataset directory path
        datasplit_folder = os.path.join(source_folder, datasplit_name, 'images')
        data_split_dest_folder = os.path.join(dest_folder, datasplit_name)
        if not os.path.exists(data_split_dest_folder):
            os.makedirs(data_split_dest_folder)

        # Check if it is a directory
        if os.path.isdir(dataset_folder):
            for file_name in os.listdir(dataset_folder):
                # Copy only .jpg photos
                if os.path.isfile(os.path.join(dataset_folder, file_name)) and file_name.endswith('.jpg'):
                    source = os.path.join(dataset_folder, file_name)
                    destination = os.path.join(data_split_dest_folder, file_name)
                    shutil.copy(source, destination)


def main():
    """ Runs data processing scripts to turn raw data from ../data/raw into
        interim data ready to make data explotation saved in ../data/interim.
    """
    # Delete the interim folder if it exists to clean creation of new data
    if os.path.exists(INTERIM_DATA_DIR):
        shutil.rmtree(INTERIM_DATA_DIR)
    if os.path.exists(INTERIM_AFFECTNET_DIR):
        shutil.rmtree(INTERIM_AFFECTNET_DIR)

    os.makedirs(INTERIM_DATA_DIR); os.makedirs(INTERIM_AFFECTNET_DIR)
    os.makedirs(os.path.join(INTERIM_DATA_DIR, "annotations"))
    os.makedirs(os.path.join(INTERIM_AFFECTNET_DIR, "annotations"))


    # Load the PAMI dataset
    mat_path = os.path.join(RAW_DATA_DIR, 'PAMI', 'annotations', 'Annotations.mat')
    mat = scipy.io.loadmat(mat_path)
    print("---------- Generating PAMI interim dataset annotations ------------")
    print("Matlab information:", mat['__header__'])

    for datasplit in mat.keys():
        if datasplit in ['test', 'train', 'val']:
            print("Working on PAMI data split:", datasplit)
            dataframe_anotations = pami_process_interim_annotations(datasplit, mat[datasplit][0])
            print("Total entries:", dataframe_anotations.shape[0])
            dataframe_anotations.to_pickle(os.path.join(INTERIM_DATA_DIR, "annotations", datasplit + '.pkl'))
    
    # Load the AffectNet dataset
    print("---------- Generating AffectNet interim dataset annotations ------------")
    for datasplit in os.listdir(RAW_AFFECTNET_DIR):
        datasplit_path = os.path.join(RAW_AFFECTNET_DIR, datasplit)
        file_list_split = os.listdir(os.path.join(datasplit_path, "annotations"))
        id_list = []
        for file in file_list_split:
            photo_idx = file.split("_")[0]
            if photo_idx not in id_list:
                id_list.append(photo_idx)
        print("Working on AffectNet data split:", datasplit)
        dataframe_anotations = affectnet_process_interim_annotations(id_list, datasplit_path)
        print("Total entries:", dataframe_anotations.shape[0])
        dataframe_anotations.to_pickle(os.path.join(INTERIM_AFFECTNET_DIR, "annotations", datasplit + '.pkl'))


    # Copy photos to interim folder
    print("---------- Copying photos to interim folder for PAMI  ------------")
    source_PAMI = os.path.join(RAW_DATA_DIR, 'PAMI', 'emotic', 'emotic')
    dest_PAMI = os.path.join(INTERIM_DATA_DIR, 'images')
    copy_images_to_interim(source_PAMI, dest_PAMI)
    print("---------- Copying photos to interim folder for AffectNet  ------------")
    
    copy_images_to_interim(source_AFFECTNET, dest_AFFECTNET)
    print("---------- Finished generating interim dataset annotations ------------")
if __name__ == '__main__':
    main()