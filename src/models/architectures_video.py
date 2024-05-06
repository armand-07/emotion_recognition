from typing import Tuple
import requests
import os

import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import torch
import albumentations
import ultralytics
import cv2

import random
import numpy as np
import wandb

from src import NUMBER_OF_EMOT, MODELS_DIR, AFFECTNET_CAT_EMOT, PROCESSED_AFFECTNET_DIR, FACE_DETECT_DIR
import src.models.architectures as arch
from src.data.dataset import data_transforms

from config import wandbAPIkey



def load_HAAR_cascade_face_detection():
    """ Loads the HAAR cascade method for face detection using the OpenCV library."""
    # Load HAAR cascade method
    try:
        detector_HAAR_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
    except Exception as e:
        print("There was a problem with HAAR_cascade using the OpenCV library:", str(e))

    return detector_HAAR_cascade



def load_HOG_SVM_cascade_face_detection():
    # Load HOG + SVM method
    try:
        detector_HOG_SVM = cv2.HOGDescriptor()
        detector_HOG_SVM.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    except Exception as e:
        print("There was a problem with HOG + SVM method using the OpenCV library:", str(e))

    return detector_HOG_SVM



def download_YOLO_model_face_recognition(size:str = "medium", directory:str = FACE_DETECT_DIR) -> None:
    """ Downloads the pretrained model for the YOLO depending on specified size. The author of the 
    weights can be found in https://github.com/akanametov/yolov8-face 
    Params:
        - size(str): Size of the model to download. Possible values: nano, medium, large
        - directory(str): Directory to save the downloaded model
    Returns:
        - None
    """
    # URL of the file to be downloaded is defined
    if size == "nano":
        url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
    elif size == "medium":
        url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-face.pt"
    elif size == "large":
        url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt"
    
    # Send a HTTP request to the URL of the file
    filename = url.split("/")[-1]
    response = requests.get(url)

    if not os.path.exists(directory):
        os.makedirs(directory)

    model_path = os.path.join(directory, filename)
    with open(model_path, mode="wb") as file:
        file.write(response.content)



def load_YOLO_model_face_recognition(device:torch.device, size:str = "medium",  directory:str = FACE_DETECT_DIR) -> ultralytics.YOLO:
    """ Loads the pretrained model for the YOLO depending on specified size. If the model is not found 
    it will be downloaded on the specified directory. The author of the weights can be found in 
    Params:
        - device (torch.device): Device to use for the model
        - size(str): Size of the model to download. Possible values: nano, medium, large
        - directory(str): Directory to load the model
    Returns:
        - model (ultralytics.YOLO): YOLO model for face recognition
    """
    assert device is not None, "Please specify the device to use for the model."
    
    model_name = None
    if size == "nano":
        model_name = "yolov8n-face.pt"
    if size == "medium":
        model_name = "yolov8m-face.pt"
    if size == "large":
        model_name = "yolov8l-face.pt"

    assert model_name is not None, "Invalid size specified. Please choose from: nano, medium, large."

    model_path = os.path.join(directory, model_name)

    # Check if the model is already downloaded
    if not os.path.exists(model_path):
        download_YOLO_model_face_recognition(size=size, directory=directory)

    # Load the model
    model = ultralytics.YOLO(model_path).to(device)
    
    return model



def load_video_models(wandb_id:str, face_detector_size:str = "medium", view_emotion_model_attention = False) -> Tuple[
    ultralytics.YOLO, torch.nn.Module, albumentations.Compose, torch.device]:
    """Function to load the models and the face detector.
    Args:
        - wandb_id (str): The id of the wandb run to be used.
        - face_detector_size (str): The size of the face detector model to be used.
    Returns:
        - face_model (ultralytics.YOLO): The face detector model.
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
    distilled_model = params['distillation']
    emotion_model, device = arch.model_creation(params['arch'], local_artifact['state_dict'])
    emotion_model.eval()
    # Load the face transforms
    face_transforms = data_transforms(only_normalize = True, image_norm = params['image_norm'], resize = True)

    # Lastly load face detector
    face_model = load_YOLO_model_face_recognition(size = face_detector_size, device = device)
    
    if view_emotion_model_attention and params['arch'].startswith('deit'):
        selected_layer = -1 # last layer, de last attention map before predicting the emotion
        emotion_model.base_model.blocks[selected_layer].attn.forward = arch.attention_forward_wrapper(emotion_model.base_model.blocks[selected_layer].attn)

    return face_model, emotion_model, distilled_model, face_transforms, device



def transform_bbox_to_square(bboxes: torch.Tensor, img_width:int, img_height:int) -> torch.Tensor:
    """Converts the input bbox to a square bbox centered in the same position as before. If the bounding box exceeds 
    the image's limit it clips it to the image's limit. The returned bbox format is [x, y, w, h], where x,y are the 
    coordinates of the top-left corner.
    Params:
        - bboxes(torch.Tensor): Bounding boxes in the format [x, y, w, h]. Expects the coordinates of the center of the bbox 
            and its width and height if the x,y were in top left. The shape is [n, 4], where n is the number of bboxes.
        - img_width(int): Width of the image
        - img_height(int): Height of the image
    Returns:
        - bboxes(torch.Tensor): Bounding boxes in the format [x, y, w, h], where the x,y represent now the top left. 
            The shape is [n, 4], where n is the number of bboxes.
    """
    max_dims = torch.max(bboxes[:, 2], bboxes[:, 3])  # Get the maximum between width and height per each detected face
    bboxes[:, 2] = max_dims  # Update width
    bboxes[:, 3] = max_dims  # Update height
    bboxes[:, 0] = bboxes[:, 0] - max_dims / 2  # Update x coordinates to top-left corner
    bboxes[:, 1] = bboxes[:, 1] - max_dims / 2  # Update y coordinates to top-left corner
    # Ensure the bbox is inside the image
    bboxes[:, 0] = torch.clamp(bboxes[:, 0], min = 0)
    bboxes[:, 1] = torch.clamp(bboxes[:, 1], min = 0)
    bboxes[:, 2] = torch.min(bboxes[:, 2], img_width - bboxes[:, 0])
    bboxes[:, 3] = torch.min(bboxes[:, 3], img_height - bboxes[:, 1])
    return bboxes



def detect_faces_HAAR_cascade(img:np.array, pretrained_model) -> np.array:
    """ Detects faces in an image using the given Haar cascade pretrained model. 
    The model returns in standard bbox format: [x, y, w, h]
    Params:
        - img: Image to detect faces
        - pretrained_model: Haar cascade model
    Returns:
        - faces_bbox (np.array): Bounding boxes of the detected faces
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform face detection
    faces_bbox = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    return faces_bbox



def detect_faces_YOLO(img:np.array, pretrained_model:ultralytics.YOLO, format:str = 'xywh', verbose:bool=False
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Detects faces in an image using the given YOLO pretrained model. The bbox is returned 
    with the specified bbox format.
    Params:
        - img (np.array): Image to detect faces
        - pretrained_model (ultralytics.YOLO): YOLO model
        - format (str): Format of the bbox returned. Options: 'xywh-center', 'xywh', 'xyxy'
        - verbose (bool): If True, it prints model information
    Returns:
        - boxes (torch.Tensor): Bounding boxes of the detected faces
        - conf (torch.Tensor): Confidence of the detection
    """
    # Make inference
    results = pretrained_model.predict(img, verbose = verbose)
    # Get confidence of detection
    conf = results[0].boxes.conf.cpu()

    # Extract the bounding boxes
    if format == 'xywh-center':
        boxes = results[0].boxes.xywh
    elif format == 'xywh':
        boxes = results[0].boxes.xywh
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2]/2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
    elif format == 'xyxy':
        boxes = results[0].boxes.xyxy
    else:
        raise ValueError('Format not supported')
    
    boxes = boxes.type(torch.int).cpu()

    return boxes, conf



def track_faces_YOLO(img:np.array, pretrained_model:ultralytics.YOLO, format:str = 'xywh', 
                     verbose:bool=False, device:torch.device = 'cuda') -> Tuple[
                         torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Detects faces in an image using the given YOLO pretrained model and uses a tracker algorithm (botsort) 
    to assign an id to the bbox that should be the same between frames. The bbox is returned with the specified
    bbox format. More info about botsort in: https://docs.ultralytics.com/modes/track/#features-at-a-glance.
    Params:
        - img (np.array): Image to detect faces
        - pretrained_model (ultralytics.YOLO): YOLO model
        - format (str): Format of the bbox returned. Options: 'xywh-center', 'xywh', 'xyxy'
        - verbose (bool): If True, it prints model information
    Returns:
        - boxes (torch.Tensor): Bounding boxes of the detected faces
        - bbox_ids (torch.Tensor): Id of the detected faces
        - conf (torch.Tensor): Confidence of the detection
    """
    # Make inference
    results = pretrained_model.track(img, verbose = verbose, persist=True, device = device) # Persist the tracking between frames botsort
    # Get confidence of detection and id of bbox
    conf = results[0].boxes.conf.cpu()
    bbox_ids = results[0].boxes.id
    if bbox_ids is None and conf.shape[0] != 0: # If no face is found it returns None
        bbox_ids =  torch.full((conf.shape[0],), -1) # If no tracking id, it returns -1
        print('Detections without tracking id, setting unknown')
    elif bbox_ids is None and conf.shape[0] == 0: # If no face is found it returns None
        bbox_ids = torch.empty(0, dtype=torch.int)
        print('No detections found')
    else:
        bbox_ids = bbox_ids.cpu().type(torch.int)
    
    # Extract the bounding boxes in specified format-
    if format == 'xywh-center':
        boxes = results[0].boxes.xywh
    elif format == 'xywh':
        boxes = results[0].boxes.xywh
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2]/2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
    elif format == 'xyxy':
        boxes = results[0].boxes.xyxy
    else:
        raise ValueError('Format not supported')
    
    boxes = boxes.type(torch.int).cpu()
        
    return boxes, bbox_ids, conf



def create_faces_batch(img:np.array, face_transforms:albumentations.Compose, 
                       face_bboxes:torch.Tensor, device:torch.device) -> torch.Tensor:
    """Create a batch of face images from the original image and a tensor of bounding boxes. 
    Bounding boxes must be inside image and be integers. Ideally they must be in 1:1 format.
    Params:
        - img (np.array): The original image.
        - face_transforms (albumentations.Compose): The face transformations.
        - face_bboxes (torch.Tensor): The tensor of bounding boxes. It has shape [D,4], where D are the detections 
            and the 4 elements are [x, y, w, h], where x,y are the coordinates of the top-left corner.
        - device (torch.device): The device to be used.
    Returns:
        - torch.Tensor: The batch of face images on specified device. It has shape [D, 3, 224, 224].
    """
    total_faces = face_bboxes.shape[0]
    face_batch = torch.zeros((total_faces, 3, 224, 224)).to(device)

    for i in range(total_faces):
        # Take face from image
        [x, y, w, h] = face_bboxes[i]
        face_img = img[y:y+h, x:x+w]
        face_batch[i] = face_transforms(image = face_img)['image']  # Apply transformations
    return face_batch



def get_raw_pred_from_frame(img:np.array, face_model:ultralytics.YOLO, emotion_model:torch.nn.Module, distilled_model:bool,
                distilled_embedding_method:str, device: torch.device, face_transforms:albumentations.Compose, 
                face_threshold:float = 0.65, tracking:bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function to infer an image or frame using the given emotion model, face detector and hyperparameters. It returns 
    the faces detected with the bbox represented in x,y top-left corner format, the predictions of the model (logits) 
    and the ids of the faces. No postprocessing or prediction normalization is done in this function.
    Args:
        - img (np.array): The image to be inferred.
        - face_model (YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - distilled_model (bool): If True, it is a distilled model. 
        - distilled_embedding_method (str): The method to obtain the output embeddings for distilled models. 
        It can be 'class', 'distill' or 'both'.
        - device (torch.device): The device to be used.
        - face_transforms (albumentations.Compose): The face transforms.
        - face_threshold (float): The threshold to be used for the face detector.
        - tracking (bool): If True, it will track the faces.
        - first_frame (bool): If True, it is the first frame of the video. It is only used for tracking
    Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The faces detected, the predictions of the model 
            (logits) and the ids of the faces.
    """
    if tracking:
        [faces_bbox, bbox_ids, confidence_YOLO] = track_faces_YOLO(img, face_model, device = device, format = 'xywh-center')
    else:
        [faces_bbox, confidence_YOLO] = detect_faces_YOLO(img, face_model, format = 'xywh-center')
        bbox_ids = torch.arange(len(faces_bbox), dtype=torch.int)

    filtered_faces = faces_bbox[confidence_YOLO > face_threshold]
    filtered_ids = bbox_ids[confidence_YOLO > face_threshold]
    
    if len(filtered_faces) != 0:
        img_height, img_width, _ = img.shape
        filtered_faces = transform_bbox_to_square(filtered_faces, img_width, img_height)
        face_batch = create_faces_batch(img, face_transforms, filtered_faces, device)
        with torch.no_grad():
            face_batch.to(device)
            if distilled_model:
                raw_preds = arch.get_pred_distilled_model(emotion_model, face_batch, distilled_embedding_method)
            else:
                raw_preds = emotion_model(face_batch) # Predict emotions returns a tensor with logits
    else: # If no faces are detected, return empty tensors
        raw_preds = torch.empty(0).to(device)
    
    return filtered_faces, raw_preds, filtered_ids



def postprocessing_inference(people_detected:dict, preds:torch.Tensor, bbox_ids:torch.Tensor, 
                             mode:str = 'standard', window_size:int = 15) -> Tuple[dict, torch.Tensor]:
    """Function to update the people_detected dictionary with the new predictions and return the output predictions based 
    on the post processing mode.
    Args:
        - people_detected (dict): The dictionary with the people detected. It has as key the id of the face 
        and as value a list of the logits of the model. It stores the window_size last logits.
        - preds (torch.Tensor): The predictions of the model. It has shape [D, NUMBER_OF_EMOT], where D are the detections.
        - bbox_ids (torch.Tensor): The ids of the faces. It has shape [D].
        - mode (str): The mode to be used for the postprocessing. It can be 'standard' or 'temporal_average'.
        - window_size (int): The size of the window to be saved in the people detected. 
    Returns:
        - dict: The updated people_detected dictionary. It has the id as key and the list of logits as value. It stores the 
            window_size last logits.
        - torch.Tensor: The updated labels list. It has shape [D, NUMBER_OF_EMOT], where D are the detections. They are logits.  
    """
    output_preds = torch.zeros((len(preds), NUMBER_OF_EMOT))
    detections = len(preds)
    for i in range(detections):
        id = bbox_ids[i].item()
        if id != -1:
            if id in people_detected:
                people_detected[id].append(preds[i])
                if len(people_detected[id]) > window_size:
                    people_detected[id] = people_detected[id][-window_size:]
            else:
                people_detected[id] = [preds[i]]
        else:
            print("No tracking id assigned to bounding box")
        # Update the output_preds
        if mode == 'standard':
            output_preds[i, :] = preds[i, :]
        elif mode == 'temporal_average':
            output_preds[i] = torch.mean(torch.stack(people_detected[id]), dim = 0) # Compute mean accross each output logit
        else:
            raise ValueError(f"Invalid mode given for postprocessing inference: {mode}")
        
    return people_detected, output_preds



def get_pred_from_frame(frame:np.array, face_model:ultralytics.YOLO, emotion_model:torch.nn.Module, device:torch.device, 
                         face_transforms:albumentations.Compose, people_detected: dict, params:dict
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Function to get the predictions from a frame using the given models and hyperparameters. It returns the faces detected
    with the bbox represented in x,y top-left corner format, the labels of the model and the ids of the faces.
    Params:
        - frame (np.array): The frame to be inferred.
        - face_model (ultralytics.YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - device (torch.device): The device to be used.
        - face_transforms (albumentations.Compose): The face transformations.
        - people_detected (dict): The dictionary with the people detected. It has as key the id of the face 
        and as value a list of the outputs of the model, depending on params it is the logits or the normalized distribution.
        - params (dict): The dictionary with the hyperparameters.
    Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]: The faces detected, the labels of the model and the 
        ids of the faces and the updated list of people detected.
    """
    faces_bbox, raw_preds, ids = get_raw_pred_from_frame(frame, face_model, emotion_model, params['distilled_model'], params['distilled_model_out_method'], 
                                                 device, face_transforms, params['face_threshold'], params['tracking'])
    if params['tracking']:
        people_detected, processed_preds = postprocessing_inference(people_detected, raw_preds, ids, 
                                                                            params['postprocessing'], params['window_size'])
    else: 
        if params['postprocessing'] != 'standard': # If tracking is disabled, only standard postprocessing is allowed
            raise ValueError(f"Invalid postprocessing mode given: {params['postprocessing']}")
        processed_preds = raw_preds
    labels = arch.get_predictions(processed_preds)
    
    return faces_bbox, labels, ids, processed_preds, people_detected