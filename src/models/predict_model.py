import argparse
import os
import time
import wandb

import cv2
import torch
import numpy as np


from src import INFERENCE_DIR, PIXELS_PER_IMAGE
import src.models.architectures as arch
from src.data.dataset import data_transforms
from src.visualization.display_annot import plot_bbox_annotations
from src.models.load_pretrained_face_models import load_YOLO_model_face_recognition, load_HAAR_cascade_face_detection
from src.models.inference_face_detection_model import detect_faces_YOLO, transform_bbox_to_square



from config import wandbAPIkey


def create_faces_batch(img, face_transforms, face_bboxes, device):
    """Create a batch of face images from the original image and a list of bounding boxes.
    Args:
        img (np.array): The original image.
        face_bboxes (np.array): An array of bounding boxes in the format [x, y, w, h].
    Returns:
        torch.Tensor: A batch of face images.
    """
    total_faces = len(face_bboxes)
    face_batch = torch.zeros((total_faces, 3, 224, 224)).to(device)
    for i in range(total_faces):
        x, y, w, h = face_bboxes[i].astype(int)
        face_img = img[y:y+h, x:x+w]
        face_batch[i] = face_transforms(image = face_img)['image']  # Apply transformations
    return face_batch



def infer_image(img, face_detector, emotion_model, device, face_transforms, face_threshold = 0.5):
    """Function to infer an image using the specified model and detector.
    Args:
        image (np.array): The image to be inferred.
        model (YOLO): The model to be used.
    Returns:
        np.array: The image with the annotations.
    """
    start = time.time()
    [faces_bbox_YOLO, confidence_YOLO] = detect_faces_YOLO(img, face_detector, format = 'xywh-center')
    end_detect_faces = time.time()
    print(f"Time to detect faces: {end_detect_faces - start}")
    
    filtered_faces = faces_bbox_YOLO[confidence_YOLO > face_threshold]
    filtered_faces = transform_bbox_to_square(filtered_faces)
    print (f"Number of faces detected: {len(filtered_faces)}")
    if len(filtered_faces) != 0:
        start_detect_emotions = time.time()
        face_batch = create_faces_batch(img, face_transforms, filtered_faces, device)
        
        with torch.no_grad():
            face_batch.to(device)
            output = emotion_model(face_batch)
            labels = arch.get_predictions(output)
            print(labels)
        end_detect_emotions = time.time()
        print(f"Time to predict emotions: {end_detect_emotions - start_detect_emotions}")
        start_plot = time.time()
        img = plot_bbox_annotations(img, filtered_faces, format = 'xywh', other_annot = labels, display = False)
        end_plot = time.time()
        print(f"Time to plot: {end_plot - start_plot}")
    else:
        print("No faces to be analyzed")
        
    print(f"Total time: {end_plot - start}")
    return img



def infer_video(cap, face_detector, emotion_model, device, face_threshold = 0.5):
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read() # ret is a boolean that returns True if the frame is available. frame is the image array.
        if ret == True:
            start = time.time()
            end = time.time()
            print(end - start)
            faces_bbox_YOLO = dict(enumerate(faces_bbox_YOLO))
            # Display the resulting frame
            cv2.imshow('Frame',frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'): # Press Q on keyboard to  exit
                break
        else: # Break the loop
            break
    # When everything done, release the video capture object
    cap.release()
 
    # Closes all the frames
    cv2.destroyAllWindows()



def main(mode: str, file: str, wandb_id: str)-> None:
    """Main function to run the inference of the model.
    Args:
        mode (str): The mode to be used for the inference.
        file (str): The file to be used for the inference.
    Returns:
        None
    """
    wandb.login(key=wandbAPIkey)
    api = wandb.Api()
    # Get emotion model
    artifact_dir = arch.get_wandb_artifact(wandb_id, api = api)
    local_artifact = torch.load(os.path.join(artifact_dir, "model_best.pt"))
    params = local_artifact["params"]
    model, device = arch.model_creation(params['arch'], local_artifact['state_dict'])
    model.eval()
    # Load the face detector
    yolo_detector = load_YOLO_model_face_recognition(size = "medium", device = device)
    face_transforms = data_transforms(only_normalize = True, image_norm = params['image_norm'], resize = True)

    # Start with inference
    if mode == 'cam':
        cap = cv2.VideoCapture(0)
    elif mode == 'video':
        path = os.path.join(INFERENCE_DIR, file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")
        cap = cv2.VideoCapture(path)
    elif mode == 'img':
        path = os.path.join(INFERENCE_DIR, file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = infer_image(img, yolo_detector, model, device, face_transforms)
        filename = os.path.join(INFERENCE_DIR, file.split('.')[0]+"inference.jpg")
        print("Saving in:", filename)
        cv2.imwrite(filename, img)
    else:
        raise ValueError(f"Invalid mode given: {mode}")
    
 

def parse_args():
    """Function to parse the arguments of the command line. It returns the arguments as a Namespace object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='img', help='The training mode to be sweep or standard run')
    parser.add_argument('--file', type=str, default= 'test1.jpg', help= 'The file to be used for the inference. If mode is cam, it is ignored. If mode is video, it is the path to the video. If mode is img, it is the path to the image.')
    parser.add_argument('--wandb_id', type=str, default='iconic-sweep-19', help='Run id to take the model weights')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    main(args.mode, args.file, args.wandb_id)