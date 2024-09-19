import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm


import pandas as pd
import numpy as np
import torch

from src import TEST_VIDEO_DIR



def generate_data(input_dir:str) -> pd.DataFrame:
    """Reads all the annotations in the input directory and generates the processed data in the output directory.
    Params:
        - input_dir (str): The path to the directory containing the annotations
        - output_dir (str): The path to the directory where the processed data will be saved
    """
    # Reading annotation
    print(f'Reading annotations from {input_dir}')
    annotations_path = os.path.join(input_dir, 'output.xml')
    # Parse XML file
    tree = ET.parse(annotations_path)
    root = tree.getroot()
    data = []

    # Iterate per each annotated frame 
    for image in root.iter('image'):
        id = image.get('name')
        filename = id.split("/")[0]
        frame = int(id.split("/")[1].split(".")[0])
        width = int(image.get('width'))
        height = int(image.get('height'))
        boxes = list(image.iter('box'))
        if not boxes: # If no bboxes information per , create an special instance to later on identify it on evaluation
            #print(f"The frame {id.split('.')[0]} has no bbox")
            label = -1
            bbox = np.array([-1, -1, -1, -1])
            data.append([filename, frame, label, bbox])
            continue

        for box in boxes: # Iterate per all the bboxes inside each image
            label = int(box.get('label').split(":")[0])
            xtl = max(int(float(box.get('xtl'))), 0)
            ytl = max(int(float(box.get('ytl'))), 0)
            xbr = min(int(float(box.get('xbr'))), width)
            ybr = min(int(float(box.get('ybr'))), height)
            bbox = np.array([xtl, ytl, xbr, ybr])
            
            data.append([filename, frame, label, bbox])
    print ("Annotations read, processing data...")
    # Create new dataframe grouping by archive and frame number to have all the bboxes and labels joined
    df = pd.DataFrame(data, columns=['filename', 'frame', 'label', 'bbox'])
    df_grouped = df.groupby(['filename', 'frame']).apply(lambda g: pd.DataFrame({
    'bboxes': [torch.Tensor(np.vstack(g['bbox'].values)).int()],
    'labels': [torch.Tensor(g['label'].values).long()]} # Convert to tensor to long to use it later on criterion
    )).reset_index()
    # Delete the columns used for the last operation
    df_grouped = df_grouped.drop(columns=['level_2'])

    return df_grouped



def main() -> None:
    """Runs data preprocessing script to turn video annotations into annotations ready to
        
    Params:
        - params (dict): The parameters to be used for the data processing
    Returns:
        - None
    """
    df = generate_data (input_dir = TEST_VIDEO_DIR)
    output_path = os.path.join(TEST_VIDEO_DIR, 'video_annotations.pkl')
    # Delete the processed folder if it exists to clean creation of new data
    if os.path.exists(output_path):
        os.remove(output_path)

    # Save the processed data
    print(f'Saving processed data to {output_path}')
    df.to_pickle(output_path)



if __name__ == '__main__':
    main()