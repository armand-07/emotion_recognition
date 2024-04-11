import argparse
import os
import time
import wandb

import cv2
import torch
import numpy as np

from src import INFERENCE_DIR
import src.models.architectures as arch
from src.data.dataset import data_transforms
from src.visualization.display_annot import plot_bbox_annotations
from src.models.load_pretrained_face_models import load_YOLO_model_face_recognition
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
    face_bboxes = face_bboxes.astype(int)
    img_height, img_width, _ = img.shape


    face_batch = torch.zeros((total_faces, 3, 224, 224)).to(device)
    for i in range(total_faces):
        # Limit to contours of image
        x = max(0, face_bboxes[i][0])
        y = max(0, face_bboxes[i][1])
        w = min(face_bboxes[i][2], img_width - x)
        h = min(face_bboxes[i][3], img_height - y)
        # Take face from image
        face_img = img[y:y+h, x:x+w]
        face_batch[i] = face_transforms(image = face_img)['image']  # Apply transformations
    return face_batch



def infer_image_debug(img, face_model, emotion_model, device, face_transforms, face_threshold = 0.5):
    """Function to infer an image using the specified model and detector.
    Args:
        image (np.array): The image to be inferred.
        model (YOLO): The model to be used.
    Returns:
        np.array: The image with the annotations.
    """
    start = time.time()
    [faces_bbox_YOLO, confidence_YOLO] = detect_faces_YOLO(img, face_model, format = 'xywh-center')
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


def infer_image(img, face_model, emotion_model, device, face_transforms, face_threshold = 0.5):
    """Function to infer an image using the specified model and detector.
    Args:
        image (np.array): The image to be inferred.
        model (YOLO): The model to be used.
    Returns:
        np.array: The image with the annotations.
    """
    [faces_bbox_YOLO, confidence_YOLO] = detect_faces_YOLO(img, face_model, format = 'xywh-center')
    
    filtered_faces = faces_bbox_YOLO[confidence_YOLO > face_threshold]
    filtered_faces = transform_bbox_to_square(filtered_faces)
    if len(filtered_faces) != 0:
        face_batch = create_faces_batch(img, face_transforms, filtered_faces, device)
        
        with torch.no_grad():
            face_batch.to(device)
            output = emotion_model(face_batch)
            labels = arch.get_predictions(output)
        img = plot_bbox_annotations(img, filtered_faces, format = 'xywh', other_annot = labels, display = False)
    else:
        print("No faces to be analyzed")
        
    return img



def infer_video_save(cap, output_cap, face_model, emotion_model, device, face_transforms, face_threshold = 0.5):
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read() # ret is a boolean that returns True if the frame is available. frame is the image array.
        if ret == True:
            frame = infer_image(frame, face_model, emotion_model, device, face_transforms, face_threshold)
            # Write the frame into the file 'output.mp4'
            output_cap.write(frame)
            #if cv2.waitKey(25) & 0xFF == ord('q'): # Press Q on keyboard to  exit
            #    break
        else: # Break the loop
            break
    # When everything done, release the video capture object
    cap.release()
    output_cap.release()



def load_models(wandb_id, face_detector_size = "medium"):
    # Load the emotion model
    wandb.login(key=wandbAPIkey)
    api = wandb.Api()
    artifact_dir = arch.get_wandb_artifact(wandb_id, api = api)
    local_artifact = torch.load(os.path.join(artifact_dir, "model_best.pt"))
    params = local_artifact["params"]
    emotion_model, device = arch.model_creation(params['arch'], local_artifact['state_dict'])
    emotion_model.eval()
    # Load the face transforms
    face_transforms = data_transforms(only_normalize = True, image_norm = params['image_norm'], resize = True)
    
    # Lastly load face detector
    face_model = load_YOLO_model_face_recognition(size = face_detector_size, device = device)
    return face_model, emotion_model, face_transforms, device



def main(mode: str, file: str, wandb_id: str, face_detector_size:str)-> None:
    """Main function to run the inference of the model.
    Args:
        mode (str): The mode to be used for the inference.
        file (str): The file to be used for the inference.
    Returns:
        None
    """
    face_model, emotion_model, face_transforms, device = load_models(wandb_id, face_detector_size = face_detector_size)
    # Start with inference
    if mode == 'cam':
        cap = cv2.VideoCapture(0)
    elif mode == 'video':
        path = os.path.join(INFERENCE_DIR, file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")
        cap = cv2.VideoCapture(path)
        # Get the width, height and fps of the video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        filename = os.path.join(INFERENCE_DIR, file.split('.')[0]+"inference.mp4")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_cap = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
        start = time.time()
        infer_video_save(cap, output_cap, face_model, emotion_model, device, face_transforms)
        end = time.time()
        print(f"The time needed to process the video: {end - start:.2f}s")
        print(f"The average time per frame: {(end - start)/total_frames:.5f}s")
        print(f"It is {((end - start)/total_frames)/(1/fps):.2f} times slower than real time inference with the video's {fps} fps")
        print("The video has been saved in:", path)

    elif mode == 'img':
        path = os.path.join(INFERENCE_DIR, file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = infer_image(img, face_model, emotion_model, device, face_transforms)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR to proper saving
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
    parser.add_argument('--face_detector_size', type=str, default='medium', help='Size of the face detector model to be used')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    main(args.mode, args.file, args.wandb_id, args.face_detector_size)