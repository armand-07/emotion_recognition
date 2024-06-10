from pathlib import Path
from prettytable import PrettyTable
from typing import Tuple
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from PIL import Image
import matplotlib.pyplot as plt
cm = plt.get_cmap('magma')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from src import INTERIM_DATA_DIR, AFFECTNET_CAT_EMOT, FROM_EMOT_TO_ID, NUMBER_OF_EMOT
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

def create_figure_mean_emotion_evolution(height:int, width:int) -> Tuple[plt.figure, plt.axis, plt.bar]:
    """ Creates a figure and axis for plotting the mean emotion evolution. 
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
    width_in = (max_length/3)/ dpi
    height_in = (max_length/4) / dpi
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)
    distribution_container = None

    return fig, ax, distribution_container



def plot_mean_emotion_distribution(img:np.array, output_preds: torch.Tensor, saving_prediction: str, fig:plt.figure, 
                                   ax:plt.axis, distribution_container:plt.bar = None)-> Tuple[np.array, plt.figure, plt.axis, plt.bar]:
    """ Plots the mean emotion distribution on the image. The distribution_container is used to update the data of the plot.
    Params:
        - img (np.array): image as a numpy array
        - output_preds (torch.Tensor): output predictions from the model
        - saving_prediction (str): The method to save the predictions. It can be 'logits' or 'distrib'.
        - fig (plt.figure): figure object
        - ax (plt.axis): axis object
        - distribution_container (plt.bar): container for the distribution plot
    Returns:
        - img (np.array): image with the mean emotion distribution plot
        - fig (plt.figure): updated figure object
        - ax (plt.axis): updated axis object
        - distribution_container (plt.bar): updated container for the distribution plot
    """
    if output_preds.shape[0] == 0: # If there are no detections, set mean_distrib to zeros
        mean_distrib = np.zeros(NUMBER_OF_EMOT)
    else:
        # Get the emotion distribution for each detection if it is in logits, else pass as it is already a distribution
        if saving_prediction == 'logits':
            output_distrib = arch.get_distributions(output_preds)
        elif saving_prediction == 'distrib':
            output_distrib = output_preds
        else:
            raise ValueError("The saving_prediction parameter must be 'logits' or 'distrib'")
        # Get mean emotion distribution across all detections
        mean_distrib = torch.mean(output_distrib, dim = 0)
        mean_distrib = mean_distrib.cpu().numpy()

    # Get the max length of the image
    height, width, _ = img.shape
    max_length = max(height, width)
    # If bar_container is None, this is the first time plotting the mean emotion distribution
    if distribution_container is None:
        distribution_container = ax.bar(AFFECTNET_CAT_EMOT, mean_distrib, color = 'cornflowerblue')
        ax.set_ylabel('Mean confidence', fontsize=int(12*(max_length/1920)))
        ax.set_ylim([0.0, 1.0])
        ax.grid(axis = 'y', linestyle = '--', linewidth = 0.5, color = 'black')
        ax.set_xlabel('Emotion categories', fontsize=int(12*(max_length/1920)))
        ax.set_title('Mean emotion confidence distribution for detections', fontsize=int(16*(max_length/1920)))
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

    # Put the distribution plot on the image
    height_distrib, width_distrib, _ = img_distrib.shape 
    padding =  int(20*(max_length/1920)) # Padding in pixels
    img[height-padding-height_distrib : height-padding, width-padding-width_distrib: width-padding] = img_distrib

    return img, fig, ax, distribution_container



def plot_mean_emotion_evolution(img:np.array, output_preds: torch.Tensor, last_mean_emotions:torch.Tensor, saving_prediction:str, 
                                EMOT_COLORS_RGB:list, fig:plt.figure, ax:plt.axis, line_container:plt.bar = None
                                )-> Tuple[np.array, plt.figure, plt.axis, plt.bar]:
    """Plots the mean emotion evolution on the image. The distribution_container is used to update the data of the plot.
    Params:
        - img (np.array): image as a numpy array
        - output_preds (torch.Tensor): output predictions from the model
        - last_mean_emotions (torch.Tensor): last mean emotions to plot. The shape is [NUMBER_OF_EMOTIONS, FRAMES_TO_PLOT]
        - saving_prediction (str): The method to save the predictions. It can be 'logits' or 'distrib'.
        - color (str): color of the line plot
        - fig (plt.figure): figure object
        - ax (plt.axis): axis object
        - line_container (plt.bar): container for the line plot
    Returns:
        - img (np.array): image with the mean emotion distribution plot
        - fig (plt.figure): updated figure object
        - ax (plt.axis): updated axis object
        - line_container (plt.bar): updated container for the line plot
    """
    if output_preds.shape[0] == 0: # If there are no detections, set mean_distrib to zeros
        mean_distrib = torch.zeros(NUMBER_OF_EMOT)
    else:
        # Get the emotion distribution for each detection if it is in logits, else pass as it is already a distribution
        if saving_prediction == 'logits':
            output_distrib = arch.get_distributions(output_preds)
        elif saving_prediction == 'distrib':
            output_distrib = output_preds
        else:
            raise ValueError("The saving_prediction parameter must be 'logits' or 'distrib'")
        # Get mean emotion distribution across all detections
        mean_distrib = torch.mean(output_distrib, dim = 0)
    
    # Update the mean emotion distribution plot
    last_mean_emotions = torch.cat((mean_distrib.view(-1, 1), last_mean_emotions[:, :-1]), dim = 1)
    numpy_mean_emotions = last_mean_emotions.cpu().numpy()
    # Get the max length of the image
    height, width, _ = img.shape
    max_length = max(height, width)
    # If bar_container is None, this is the first time plotting the mean emotion distribution
    if line_container is None:
        matplotlib_colors = [(float(r / 255), float(g / 255), float(b / 255)) for r, g, b in EMOT_COLORS_RGB]
        line_container = []
        for i in range(numpy_mean_emotions.shape[0]):
            line, = ax.plot(numpy_mean_emotions[i], color=matplotlib_colors[i])
            line_container.append(line)
        ax.set_ylabel('Mean confidence', fontsize=int(12*(max_length/1920)))
        ax.set_ylim([0.0, 1.0])
        ax.grid(axis = 'y', linestyle = '--', linewidth = 0.5, color = 'black')
        ax.set_xlabel('Past frames', fontsize=int(12*(max_length/1920)))
        ax.set_xlim([0, numpy_mean_emotions.shape[1]-1])
        ax.invert_xaxis()
        ax.set_title('Mean emotion confidence evolution for detections', fontsize=int(16*(max_length/1920)))
        ax.legend(line_container, AFFECTNET_CAT_EMOT, loc='upper left', fontsize=int(12*(max_length/1920)))
        # Set the font size for the tick labels
        plt.xticks(fontsize=int(9*(max_length/1920)))
        plt.yticks(fontsize=int(9*(max_length/1920)))
        plt.subplots_adjust(left = 0.15, right = 0.95, top=0.9, bottom=0.15)
    else: # If bar_container is not None just update the data of the plot
        for i, new_data in enumerate(numpy_mean_emotions): # Update the data of the plot per emotion
            line_container[i].set_ydata(new_data)

    # Convert plot to image as a numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_plot = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (3,))

    # Put the distribution plot on the image
    height_plot, width_plot, _ = img_plot.shape 
    padding =  int(20*(max_length/1920)) # Padding in pixels
    img[height-padding-height_plot : height-padding, width-padding-width_plot: width-padding] = img_plot

    return img, last_mean_emotions, fig, ax, line_container



def plot_bbox_emot(img:np.array, bbox:np.array, labels:list, bbox_ids:np.array = None, cls_weight:torch.Tensor = None,  bbox_format:str ="xywh", 
                   display:bool = True, color_list:list = None) -> np.array:
    """ Displays the bounding boxes and emotion annotations
    """
    img_height, img_width, _ = img.shape
    max_img_size = max(img_height, img_width)
    # Define the parameters for the rectangle
    if color_list is None:
        bbox_color = (0, 255, 0) # Green color in RGB
    text_color = (0, 0, 0) # Black color in RGB

    for i in range(len(bbox)):
        if bbox_format == "xywh":
            x, y, w, h = bbox[i].tolist()

        elif bbox_format == "xywh-center":
            x_center, y_center, w, h = bbox[i].tolist()
            x = x_center-int(w/2)
            y = y_center-int(h/2)
        elif bbox_format == "xyxy":
            [x, y, x2, y2] = bbox[i].tolist()
            w = x2 - x
            h = y2 - y
        else:
            raise ValueError("The format of the bounding box is not valid. It must be 'xywh', 'xywh-center' or 'xyxy'")
        id = bbox_ids[i].item()
        if id != -1 and bbox_ids is not None: # If bbox_ids is not None and the id is not Unknown
            text = str(id) + ":" + labels[i]
            if color_list is not None:
                bbox_color = color_list[FROM_EMOT_TO_ID[labels[i]]]
        elif id == -1: # If bbox_ids is not None and the id is Unknown
            text = 'Unknown tracking' # Unknown detection as the bbox_id is -1
            bbox_color = (128, 128, 128) # Grey color in RGB
        else: # If bbox_ids is None
            text = str(int(i))+":"+labels[i]
            if color_list is not None:
                bbox_color = color_list[FROM_EMOT_TO_ID[labels[i]]]

        # If cls_weight is not None, it will display the class attention map with what the emotion model is seeing
        if cls_weight is not None:
            max_size = max(w, h)
            # Correct x_plot and y_plot to avoid going out of the image on the left and top (if corrected is not 1:1)
            x_plot = int((x+w/2) - max_size/2)
            y_plot = int((y+h/2) - max_size/2)
            w_plot = max_size + min(x_plot, 0)
            h_plot = max_size + min(y_plot, 0)
            x_plot = max(x_plot, 0)
            y_plot = max(y_plot, 0)
            # Correct h_plot and w_plot to avoid going out of the image on the right and bottom (if corrected is not 1:1)
            w_plot = min(w_plot, img_width - x_plot)
            h_plot = min(h_plot, img_height - y_plot)

            # Resize the 14x14 class attention map to the plot size using bilinear interpolation
            cls_resized = F.interpolate(cls_weight[i].view(1, 1, 14, 14), (h_plot, w_plot), 
                                        mode='bilinear').view(h_plot, w_plot).numpy()
            # Normalize cls_resized to the range [0, 1] to properly apply the color map
            cls_resized_np = (cls_resized - cls_resized.min()) / (cls_resized.max() - cls_resized.min())
            # Apply the magma color map
            cls_colored = cm(cls_resized_np, bytes=True)
            # Return to RGB format
            cls_colored = cv2.cvtColor(cls_colored, cv2.COLOR_RGBA2RGB)
            # Blend the colored class attention map with the image
            img[y_plot:y_plot+h_plot, x_plot:x_plot+w_plot] = cv2.addWeighted(
                img[y_plot:y_plot+h_plot, x_plot:x_plot+w_plot], 0.55, cls_colored, 0.45, 0)
            
        # First set the thickness of the bbox
        bbox_thickness = int(round(max_img_size / 500))
        bbox_thickness = max(1, bbox_thickness)
        # Add bbox
        cv2.rectangle(img, (x, y), (x+w, y+h), bbox_color, bbox_thickness)
            
        # Finds space required by the text so that we can put a correct background
        font_scale = max_img_size / 1500
        text_thickness = int(round(max_img_size / 750))
        text_thickness = max(1, text_thickness) # Make sure text_thickness is at least 1
        (w_text, h_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX , font_scale, text_thickness) 

        # Add text and background above the bbox
        img = cv2.rectangle(img, (x, y - int(h_text*1.5)), (x + w_text, y), bbox_color, -1)
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
    sample_path = os.path.join(INTERIM_DATA_DIR, 'images', Path(sample_df['path']))

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