import argparse
import os
import time
import wandb
from tqdm import tqdm
from typing import Union, Tuple

import cv2
import torch
import numpy as np
import albumentations

from src import INFERENCE_DIR, NUMBER_OF_EMOT
import src.models.architectures as arch
from src.data.dataset import data_transforms
from src.visualization.display_annot import plot_bbox_emot
from src.models.load_pretrained_face_models import load_YOLO_model_face_recognition
from src.models.inference_face_detection_model import detect_faces_YOLO, track_faces_YOLO, transform_bbox_to_square

from config import wandbAPIkey



def create_faces_batch(img, face_transforms, face_bboxes, device) -> torch.Tensor:
    """Create a batch of face images from the original image and a list of bounding boxes. 
    Bounding boxes must be inside image and be integers. Ideally they must be in 1:1 format
    Args:
        - img (np.array): The original image.
        - face_transforms (albumentations.Compose): The face transformations.
        - face_bboxes (np.array): The list of bounding boxes.
        - device (torch.device): The device to be used.
    Returns:
        - torch.Tensor: The batch of face images on specified device.
    """
    total_faces = face_bboxes.shape[0]
    face_batch = torch.zeros((total_faces, 3, 224, 224)).to(device)

    for i in range(total_faces):
        # Take face from image
        [x, y, w, h] = face_bboxes[i]
        face_img = img[y:y+h, x:x+w]
        face_batch[i] = face_transforms(image = face_img)['image']  # Apply transformations
    return face_batch



def infer_image(img:np.array, face_model, emotion_model:torch.nn.Module, device: torch.device, 
                face_transforms:albumentations.Compose, face_threshold:float = 0.5) -> Tuple[np.array, torch.Tensor, np.array]:
    """Function to infer an image using the specified model and detector.
    Args:
        - img (np.array): The image to be inferred.
        - face_model (YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - device (torch.device): The device to be used.
        - face_transforms (albumentations.Compose): The face transforms.
        - face_threshold (float): The threshold to be used for the face detector.
    Returns:
        - 
    """
    track = True
    if track:
        [faces_bbox, bbox_ids, confidence_YOLO] = track_faces_YOLO(img, face_model, format = 'xywh-center')
    else:
        [faces_bbox, confidence_YOLO] = detect_faces_YOLO(img, face_model, format = 'xywh-center')
        bbox_ids = torch.Tensor(range(len(faces_bbox)))

    filtered_faces = faces_bbox[confidence_YOLO > face_threshold]
    filtered_ids = bbox_ids[confidence_YOLO > face_threshold]
    
    if len(filtered_faces) != 0:
        img_height, img_width, _ = img.shape
        filtered_faces = transform_bbox_to_square(filtered_faces, img_width, img_height)
        face_batch = create_faces_batch(img, face_transforms, filtered_faces, device)
        
        with torch.no_grad():
            face_batch.to(device)
            preds = emotion_model(face_batch) # Predict emotions returns a tensor with logits
    else:
        print("No faces to be analyzed")
        preds = torch.empty(0).to(device)
    
    return filtered_faces, preds, filtered_ids



def postprocessing_inference(people_detected, preds, ids, mode = 'standard', window_size = 15) -> Tuple[dict, torch.Tensor]:
    """Function to update the people_detected dictionary with the new labels.
    Args:
        - people_detected (dict): The dictionary with the people detected.
        - labels (list): The list of labels detected.
    Returns:
        
    """
    output_preds = torch.zeros((len(preds), NUMBER_OF_EMOT))
    for i in range(len(preds)):
        if ids[i] in people_detected:
            people_detected[ids[i]].append(preds[i])
            if len(people_detected[ids[i]]) > window_size:
                people_detected[ids[i]] = people_detected[ids[i]][-window_size:]
        else:
            people_detected[ids[i]] = [preds[i]]
        # Update the output_preds
        if mode == 'standard':
            output_preds[i, :] = preds[i, :]
        elif mode == 'temporal_average':
            output_preds[i] = torch.mean(torch.stack(people_detected[ids[i]]), dim = 0) # Compute mean accross each output logit
        else:
            raise ValueError(f"Invalid mode given for postprocessing inference: {mode}")

    return people_detected, output_preds



def infer_stream(cap:cv2.VideoCapture, face_model, emotion_model: torch.nn.Module, device: torch.device, 
                face_transforms:albumentations.Compose, face_threshold:float = 0.5) -> None:
    """Function to make inference in a video stream. It shows the results in a window.
    Args:
        - cap (cv2.VideoCapture): The video capture object.
        - face_model (YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - face_transforms (albumentations.Compose): The face transforms.
        - face_threshold (float): The threshold to be used for the face detector.
    Returns:
        - None"""
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            faces_bbox, labels, ids = infer_image(frame, face_model, emotion_model, device, face_transforms, face_threshold)
            postprocessing_inference(faces_bbox, labels, ids)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): # Press Q on keyboard to exit
                break
        else:
            break



def infer_video_save(cap: cv2.VideoCapture, output_cap: cv2.VideoWriter, name:str, face_model, 
                     emotion_model: torch.nn.Module, device: torch.device, face_transforms: albumentations.Compose, 
                     face_threshold = 0.5) -> None:
    """Function to make inference in a video and save the result in capturer.
    Args:
        - cap (cv2.VideoCapture): The video capture object.
        - output_cap (cv2.VideoWriter): The video writer object.
        - name (str): The name of the video.
        - face_model (YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - device (torch.device): The device to be used.
        - face_transforms (albumentations.Compose): The face transforms.
        - face_threshold (float): The threshold to be used for the face detector.
    Returns:
        - None
    """
    # Read until video is completed
    tracking = True 
    if tracking:
        people_detected = dict()

    for frame_id in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc = f'Analizing video "{name}"'):
        # Capture frame-by-frame
        ret, frame = cap.read() # ret is a boolean that returns True if the frame is available. frame is the image array.
        if ret == True:
            faces_bbox, preds, ids = infer_image(frame, face_model, emotion_model, device, face_transforms, face_threshold)
            ids = ids.cpu().numpy()
            faces_bbox = faces_bbox.cpu().numpy()
            if tracking:
                mode = 'temporal_average'
                people_detected, output_preds = postprocessing_inference(people_detected, preds, ids, mode, window_size=15)
            labels = arch.get_predictions(output_preds)
            frame = plot_bbox_emot(frame, faces_bbox, labels, ids, bbox_format ="xywh", display = False)
            # Write the frame into the output file
            output_cap.write(frame)
        else: # Break the loop
            break



def load_models(wandb_id:str, face_detector_size:str = "medium") -> Tuple[
    torch.nn.Module, torch.nn.Module, albumentations.Compose, torch.device]:
    """Function to load the models and the face detector.
    Args:
        - wandb_id (str): The id of the wandb run to be used.
        - face_detector_size (str): The size of the face detector model to be used.
    Returns:
        - face_model (YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - face_transforms (albumentations.Compose): The face transforms.
        - device (torch.device): The device to be used.
    """
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



def process_file(input_path:str, output_dir:str, face_model, emotion_model, device, face_transforms) -> None:
    """Function to process a file, either an image or a video. It saves the results in the output_dir.
    Args:
        - input_path (str): The path to the file to be processed.
        - output_dir (str): The directory to save the results.
        - face_model (YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - device (torch.device): The device to be used.
        - face_transforms (albumentations.Compose): The face transforms.
    Returns:
        - None"""
    if input_path.endswith('.jpg') or input_path.endswith('.png'):
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        start = time.time()
        img = infer_image(img, face_model, emotion_model, device, face_transforms)
        end = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR to proper saving
        # Define the output path
        name = os.path.basename(input_path).split('.')[0]
        output_filename = os.path.join(output_dir, name+"_inference.jpg")
        if os.path.exists(output_filename):
            os.remove(output_filename)
        # Save the image
        cv2.imwrite(output_filename, img)
        print("Saving in:", output_filename)
        print(f"The time needed to process the photo: {end - start:.2f}s")

    elif input_path.endswith('.mp4') or input_path.endswith('.avi') or input_path.endswith('.mov'):
        cap = cv2.VideoCapture(input_path)
        # Get the width, height, total frames and fps of the video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the output path
        name = os.path.basename(input_path).split('.')[0]
        output_filename = os.path.join(output_dir, name+"_inference.mp4")
        if os.path.exists(output_filename):
            os.remove(output_filename)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_cap = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
        
        # Make inference
        start = time.time()
        infer_video_save(cap, output_cap, name, face_model, emotion_model, device, face_transforms)
        end = time.time()
        # Store results and release the video capture object
        cap.release()
        output_cap.release()
        print(f"The time needed to process the video: {end - start:.2f}s")
        print(f"The average time per frame: {(end - start)/total_frames:.5f}s")
        print(f"It is {((end - start)/total_frames)/(1/fps):.2f} times slower than real time inference with the video's {fps} fps")
        print("The video has been saved in:", output_filename)
    
    else:
        raise ValueError(f"Invalid archive type given: {os.path.basename(input_path).split('.')[1]}")
    


def main(mode: str, input_path: str, output_dir:str, wandb_id: str, face_detector_size:str) -> None:
    """Main function to run the inference of the model.
    Args:
        mode (str): The mode to be used for the inference.
        file (str): The file to be used for the inference.
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    face_model, emotion_model, face_transforms, device = load_models(wandb_id, face_detector_size = face_detector_size)
    # Start with inference
    if mode == 'stream':
        cap = cv2.VideoCapture(0)

    elif mode == 'save':
        input_path = os.path.join(INFERENCE_DIR, input_path)
        if os.path.isdir(input_path):
            print(f"Found a directory")
            files = os.listdir(input_path)
            for file in files:
                input_file_path = os.path.join(input_path, file)
                process_file(input_file_path, output_dir, face_model, emotion_model, device, face_transforms)
        elif os.path.isfile(input_path):
            print(f"Found an archive")
            process_file(input_path, output_dir, face_model, emotion_model, device, face_transforms)
        else:
            print(f"{input_path} is neither a directory nor an archive file")

    else:
        raise ValueError(f"Invalid mode given: {mode}")
    


def parse_args():
    """Function to parse the arguments of the command line. It returns the arguments as a Namespace object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='save', help='Process to stream results from camera, or make inference to saved img or video')
    parser.add_argument('--input_path', type=str, default = 'test', help= 'The file to be used for the inference. If mode is cam, it is ignored. If mode is video or img, it is the path to the archive.')
    parser.add_argument('--output_dir', type=str, default = os.path.join(INFERENCE_DIR, 'output'), help= 'Directory to save results')
    parser.add_argument('--wandb_id', type=str, default='iconic-sweep-19', help='Run id to take the model weights')
    parser.add_argument('--face_detector_size', type=str, default='medium', help='Size of the face detector model to be used')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    main(args.mode, args.input_path, args.output_dir, args.wandb_id, args.face_detector_size)