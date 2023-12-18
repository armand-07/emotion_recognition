import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
from src import INTERIM_DATA_DIR
from prettytable import PrettyTable
from random import randint
from PIL import Image


def display_img_annot (sample_df, bbox_thickness = 2, font_size = 0.6):
    sample_path = Path(os.path.join(INTERIM_DATA_DIR,'images', sample_df['path']))
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
    annotations_table.field_names = ["Person","Gender", "Age", "Emotions categories", "Continious emotions (Valence,Arousal,Dominance)"]

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
        # Finds space required by the text so that we can put a background with that amount of width.
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
        path = Path(os.path.join(INTERIM_DATA_DIR,'images', images[i]))
        img = Image.open(str(path))
        axs[i].imshow(img)

    plt.tight_layout()
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=0.95)
    plt.show()