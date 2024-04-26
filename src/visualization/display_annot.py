from pathlib import Path
from prettytable import PrettyTable
from typing import Tuple

import cv2
import numpy as np
import torch
from random import randint
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from src import INTERIM_DATA_DIR, AFFECTNET_CAT_EMOT
import src.models.architectures as arch



def create_figure_mean_emotion_distribution(height:int, width:int) -> Tuple[plt.figure, plt.axis, plt.bar]:
    """ Creates a figure and axis for plotting the mean emotion distribution. 
    The distribution_container defined as None is used to update the data of the plot.
    Params:
        - height (int): height of the image
        - width (int): width of the image
    Returns:
        - fig (plt.figure): figure object
        - ax (plt.axis): axis object
        - distribution_container (plt.bar): container for the distribution plot"""
    # Precompute the plot size depending on image size
    max_length = max(height, width)
    dpi = 100 # Dots per inch, standard is 100
    width_in = (max_length/4)/ dpi
    height_in = (max_length/4) / dpi
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    distribution_container = None

    return fig, ax, distribution_container



def plot_mean_emotion_distribution(img:np.array, output_preds: torch.Tensor, fig:plt.figure, 
                                   ax:plt.axis, distribution_container:plt.bar = None)-> Tuple[np.array, plt.figure, plt.axis, plt.bar]:
    """ Plots the mean emotion distribution on the image. The distribution_container is used to update the data of the plot.
    Params:
        - img (np.array): image as a numpy array
        - output_preds (torch.Tensor): output predictions from the model
        - fig (plt.figure): figure object
        - ax (plt.axis): axis object
        - distribution_container (plt.bar): container for the distribution plot
    Returns:
        - img (np.array): image with the mean emotion distribution plot
        - fig (plt.figure): updated figure object
        - ax (plt.axis): updated axis object
        - distribution_container (plt.bar): updated container for the distribution plot
    """
    # Get the emotion distribution for each detection
    output_distrib = arch.get_distributions(output_preds)
    # Get mean emotion distribution across all detections
    mean_distrib = torch.mean(output_distrib, dim = 0)
    mean_distrib = mean_distrib.cpu().numpy()

    # Get the max length of the image
    height, width, _ = img.shape
    max_length = max(height, width)
    # If bar_container is None, this is the first time plotting the mean emotion distribution
    if distribution_container is None:
        distribution_container = ax.bar(AFFECTNET_CAT_EMOT, mean_distrib, color = 'blue')
        ax.set_ylabel('Probability', fontsize=int(12*(max_length/1920)))
        ax.set_ylim([0.0, 1.0])
        ax.grid(axis = 'y', linestyle = '--', linewidth = 0.5, color = 'black')
        ax.set_xlabel('Emotion categories', fontsize=int(12*(max_length/1920)))
        ax.set_title('Mean emotion distribution for detections', fontsize=int(16*(max_length/1920)))
        # Set the font size for the tick labels
        plt.xticks(fontsize=int(9*(max_length/1920)))
        plt.yticks(fontsize=int(9*(max_length/1920)))
        plt.subplots_adjust(left = 0.15, right = 0.95, top=0.9, bottom=0.15)
    else: # If bar_container is not None just update the data of the plot
        for bar, h in zip(distribution_container, mean_distrib):
            bar.set_height(h)

    # Convert plot to image as a numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_distrib = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (3,))
    img_distrib = cv2.cvtColor(img_distrib, cv2.COLOR_RGB2BGR)

    # Put the distribution plot on the image
    height_distrib, width_distrib, _ = img_distrib.shape 
    padding =  int(20*(max_length/1920)) # Padding in pixels
    img[height-padding-height_distrib : height-padding, width-padding-width_distrib: width-padding] = img_distrib

    return img, fig, ax, distribution_container



def plot_bbox_emot(img:np.array, bbox:np.array, labels:list, bbox_ids:np.array = None, bbox_format:str ="xywh", 
                   display:bool = True) -> np.array:
    """ Displays the bounding boxes and emotion annotations
    """
    height, width, _ = img.shape
    max_img_size = max(height, width)
    # Define the parameters for the rectangle
    background_color = (0, 255, 0) # Green color in RGB
    text_color = (0, 0, 0) # White color in RGB

    for i in range(len(bbox)):
        if bbox_format == "xywh":
            x, y, w, h = bbox[i].tolist()

        elif bbox_format == "xywh-center":
            x, y, w, h = bbox[i].tolist()
            x = x-int(w/2); 
            y = y-int(h/2)
        elif bbox_format == "xyxy":
            [x, y, x2, y2] = bbox[i].tolist()
            w = x2 - x
            h = y2 - y
        else:
            raise ValueError("The format of the bounding box is not valid. It must be 'xywh', 'xywh-center' or 'xyxy'")
        id = bbox_ids[i].item()
        if id != -1 and bbox_ids is not None:
            text = str(id)+":"+labels[i]
        elif id == -1:
            text = 'Unknown' # Unknown detection as the bbox_id is -1
        else:
            text = str(i)+":"+labels[i]
            
        # First set the thickness of the bbox
        bbox_thickness = int(round(max_img_size / 500))
        bbox_thickness = max(1, bbox_thickness)
        # Add bbox
        cv2.rectangle(img, (x, y), (x+w, y+h), background_color, bbox_thickness)
            
        # Finds space required by the text so that we can put a correct background
        font_scale = max_img_size / 1500
        text_thickness = int(round(max_img_size / 750))
        text_thickness = max(1, text_thickness) # Make sure text_thickness is at least 1
        (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX , font_scale, text_thickness) 

        # Add text and background above the bbox
        img = cv2.rectangle(img, (x, y - int(h_text*1.5)), (x + w_text, y), background_color, -1)
        img = cv2.putText(img, text, (x, y - int(h_text*0.3)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, int(text_thickness), lineType = cv2.LINE_AA)
    if display:
        # Displaying the image  
        print("Displaying the image")
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        return img



def plot_bbox_annotations(img, bbox_annot, format ="xywh", conf_threshold = 0, conf = None,  other_annot = None, display = True):
    """ Displays the bounding boxes and text annotations on the image. The standard format is xywh
    """
    height, width, _ = img.shape
    max_img_size = max(height, width)
    # Define the parameters for the rectangle
    background_color = (0, 255, 0) # Green color in RGB
    text_color = (0, 0, 0) # White color in RGB

    for i in range(len(bbox_annot)):
        if conf is not None:
            annot_conf = conf[i]
        else:
            annot_conf = 1
        if annot_conf > conf_threshold:
            if format == "xywh":
                [x, y, w, h] = bbox_annot[i].astype(int)
            elif format == "xywh-center":
                [x, y, w, h] = bbox_annot[i].astype(int)
                x = x-int(w/2); 
                y = y-int(h/2)
            elif format == "xyxy":
                [x, y, x2, y2] = bbox_annot[i].astype(int)
                w = x2 - x
                h = y2 - y
            else:
                raise ValueError("The format of the bounding box is not valid. It must be 'xywh', 'xywh-center' or 'xyxy'")
            if other_annot is not None:
                text = other_annot[i]
            else:
                text = ""
            text = str(i)+":"+text

            # First set the thickness of the bbox
            bbox_thickness = int(round(max_img_size / 500))
            bbox_thickness = max(1, bbox_thickness)
            # Add bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), background_color, bbox_thickness)
            
            
            # Finds space required by the text so that we can put a correct background
            font_scale = max_img_size / 1500
            text_thickness = int(round(max_img_size / 750))
            text_thickness = max(1, text_thickness) # Make sure text_thickness is at least 1
            (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX , font_scale, text_thickness) 

            # Add text and background above the bbox
            img = cv2.rectangle(img, (x, y - int(h_text*1.5)), (x + w_text, y), background_color, -1)
            img = cv2.putText(img, text, (x, y - int(h_text*0.3)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, int(text_thickness), lineType = cv2.LINE_AA)
    if display:
        # Displaying the image  
        print("Displaying the image")
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        return img



def display_img_annot_PAMI (sample_df, bbox_thickness = 2, font_size = 0.6):
    sample_path = Path(sample_df['path'])
    print("The path of the example image is:", sample_path)
    print("The image orig DB is:", sample_df['orig_db'])

    # Reading an image and transforming it to RGB
    img = cv2.imread(str(sample_path)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Convert BGR to RGB
    print("The image shape is:", img.shape)
    print("There is a total of", sample_df['people'], "annotated people in the image")

    # Define the parameters for the rectangle
    text_color = (255, 255, 255)                    # White color in RGB



    # Create a table
    annotations_table = PrettyTable()
    # Define column names
    annotations_table.field_names = ["Person","Gender", "Age", "Emotions categories", "Continuous emotions [Valence, Arousal, Dominance]"]

    for person_id in range(sample_df['people']):
        # Window characteristics
        r = randint(50, 150); g = randint(50, 150); b = randint(50, 150)
        color = (r, g, b)       # Blue color in RGB
        x1 = sample_df['bbox'][person_id][0]; y1 = sample_df['bbox'][person_id][1]
        x2 = sample_df['bbox'][person_id][2]; y2 = sample_df['bbox'][person_id][3]

        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > img.shape[1]: x2 = img.shape[1]
        if y2 > img.shape[0]: y2 = img.shape[0]

        # Window name in which image is displayed
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, bbox_thickness) 
        
        # For the text background
        # Finds space required by the text so that we can put a background with that amount of max_img_size.
        (w, h), _ = cv2.getTextSize(str(person_id), cv2.FONT_HERSHEY_DUPLEX , font_size, int(bbox_thickness*0.5))
        # Prints the text.    
        img = cv2.rectangle(img, (x1, y1 - int(h*1.5)), (x1 + w, y1), color, -1)
        img = cv2.putText(img, str(person_id), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_size, (text_color), int(bbox_thickness*0.5))
        
        annotations_table.add_row([person_id, sample_df['gender'][person_id][:], sample_df['age'][person_id], 
                                   sample_df['label_cat'][person_id], sample_df['label_cont'][person_id]])
    
    # Displaying the image  
    plt.imshow(img)  
    plt.axis('off')
    plt.show()

    # Displaying the table
    print(annotations_table)


    
def display_img_set (grid_size, images, title = None):
    # Assuming images is a list of file paths to your images
    assert len(images) > 0, "No images to display."
    assert type(images) == list, "Images must be a list of file paths."

    num_images = min(len(images), grid_size[0]*grid_size[1])
    _, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[0]*2, grid_size[1]*2),
                                subplot_kw={'xticks': [], 'yticks': []})

    axs = axs.ravel()

    for i in range(num_images):
        path = Path(images[i])
        img = Image.open(str(path))
        axs[i].imshow(img)

    plt.tight_layout()
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=0.95)
    plt.show()