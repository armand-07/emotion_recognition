import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
from src import INTERIM_DATA_DIR
from prettytable import PrettyTable
from random import randint


def display_img_annot (row_example):
    row_example_path = Path(os.path.join(INTERIM_DATA_DIR,'images', row_example['path']))
    print("The path of the example image is:", row_example_path)
    print("There is a total of", row_example['people'], "annotated people in the image")

    # Reading an image and transforming it to RGB
    img = cv2.imread(str(row_example_path)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Convert BGR to RGB
    thickness = 2                                   # Line thickness of 2 px
    text_color = (255, 255, 255)                    # White color in RGB

    # Create a table
    annotations_table = PrettyTable()
    # Define column names
    annotations_table.field_names = ["Person","Gender", "Age", "Emotions categories", "Continious emotions (Valence,Arousal,Dominance)"]

    for person_id in range(row_example['people']):
        # Window characteristics
        r = randint(50, 150); g = randint(50, 150); b = randint(50, 150)
        color = (r, g, b)       # Blue color in RGB
        x1 = row_example['bbox'][person_id][0]; y1 = row_example['bbox'][person_id][1]
        x2 = row_example['bbox'][person_id][2]; y2 = row_example['bbox'][person_id][3]

        # Window name in which image is displayed
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness) 
        
        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        (w, h), _ = cv2.getTextSize(str(person_id), cv2.FONT_HERSHEY_DUPLEX , 0.6, 1)

        # Prints the text.    
        img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        img = cv2.putText(img, str(person_id), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (text_color), 1)
        
        annotations_table.add_row([person_id, row_example['gender'][person_id][:], row_example['age'][person_id], 
                                   row_example['label_cat'][person_id], row_example['label_cont'][person_id]])
    
    # Displaying the image  
    plt.imshow(img)  
    plt.axis('off')
    plt.show()

    # Displaying the table
    print(annotations_table)

    
