from typing import Tuple
import requests
import os

import torch
import albumentations
import ultralytics
import cv2

import numpy as np
import wandb

from src import NUMBER_OF_EMOT, FACE_DETECT_DIR
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
    weights can be found in: https://github.com/akanametov/yolov8-face 
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
    it will be downloaded on the specified directory. The author of the weights can be found in:
    https://github.com/akanametov/yolov8-face. If tracking is enabled during inference, the model needs 
    to be reloaded when the inference is done in different videos. 
    Params:
        - device (torch.device): Device to use for the model
        - size(str): Size of the model to download. Possible values: nano, medium, large
        - directory(str): Directory to load the model
    Returns:
        - model (ultralytics.YOLO): YOLO model for face recognition
    """
    assert device is not None, "Please specify the device to use for the model."
    print("Loading YOLO trained model for face recognition with size: ", size)
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
    print("The face model is loaded with CUDA:", ultralytics.utils.checks.cuda_is_available())
    model = ultralytics.YOLO(model_path).to(device)
    
    return model



def load_video_models(wandb_id:str, face_detector_size:str = "medium", view_emotion_model_attention:bool = False, 
                      cpu_device:bool = False) -> Tuple[ultralytics.YOLO, torch.nn.Module, albumentations.Compose, torch.device]:
    """Function to load the models and the face detector.
    Params:
        - wandb_id (str): The id of the wandb run to be used.
        - face_detector_size (str): The size of the face detector model to be used.
        - view_emotion_model_attention (bool): If True, it will save the attention maps of the emotion model when forwarding the images.
        - cpu_device (bool): If True, it will use only the CPU device.
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
    if torch.cuda.is_available():
        map_location = None
    else:
        map_location = torch.device('cpu')
    local_artifact = torch.load(os.path.join(artifact_dir, "model_best.pt"), map_location=map_location)
    params = local_artifact["params"]
    if 'distillation' in params:
        distilled_model = params['distillation']
    else:
        distilled_model = False
    if cpu_device:
        device = torch.device('cpu')
        emotion_model, _ = arch.model_creation(params['arch'], local_artifact['state_dict'], device)
    else:
        emotion_model, device = arch.model_creation(params['arch'], local_artifact['state_dict'])
        torch.cuda.set_device("cuda:0")
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
    the image's limit it clips it to the image's limit (so the bbox is no longer 1:1). The returned bbox format is [x, y, w, h], 
    where x,y are the coordinates of the top-left corner.
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
    squared_bboxes = torch.zeros_like(bboxes)
    squared_bboxes[:, 2] = max_dims  # Update width
    squared_bboxes[:, 3] = max_dims  # Update height
    squared_bboxes[:, 0] = bboxes[:, 0] - max_dims / 2  # Update x coordinates to top-left corner
    squared_bboxes[:, 1] = bboxes[:, 1] - max_dims / 2  # Update y coordinates to top-left corner
    # Ensure the bbox is inside the left and top limits, subtracts the negative values to width and height (so bbox is not 1:1 on limits)
    squared_bboxes[:, 2] = squared_bboxes[:, 2] + torch.clamp(squared_bboxes[:, 0], max = 0)
    squared_bboxes[:, 3] = squared_bboxes[:, 3] + torch.clamp(squared_bboxes[:, 1], max = 0)
    squared_bboxes[:, 0] = torch.clamp(squared_bboxes[:, 0], min = 0)
    squared_bboxes[:, 1] = torch.clamp(squared_bboxes[:, 1], min = 0)
    # Ensure the bbox is inside the right and bottom limits of the image (so bbox is not 1:1 on limits)
    squared_bboxes[:, 2] = torch.clamp(squared_bboxes[:, 2], max = img_width - squared_bboxes[:, 0])
    squared_bboxes[:, 3] = torch.clamp(squared_bboxes[:, 3], max = img_height - squared_bboxes[:, 1])

    return squared_bboxes



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



def detect_faces_YOLO(img:np.array, pretrained_model:ultralytics.YOLO, format:str = 'xywh', verbose:bool=False,
                      device:torch.device = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """ Detects faces in an image using the given YOLO pretrained model. The bbox is returned 
    with the specified bbox format.
    Params:
        - img (np.array): Image to detect faces
        - pretrained_model (ultralytics.YOLO): YOLO model
        - format (str): Format of the bbox returned. Options: 'xywh-center', 'xywh', 'xyxy'
        - verbose (bool): If True, it prints model information
        - device (torch.device): Device to use for the model
    Returns:
        - boxes (torch.Tensor): Bounding boxes of the detected faces
        - conf (torch.Tensor): Confidence of the detection
    """
    # Make inference
    results = pretrained_model.predict(img, verbose = verbose, device = device)
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
        - bbox_ids (torch.Tensor): Id of the detected faces. The track ids starts in id 1.
        - conf (torch.Tensor): Confidence of the detection
    """
    results = pretrained_model.track(img, verbose = verbose, persist=True, device = device) # Persist the tracking between frames botsort
    # Get confidence of detection and id of bbox
    conf = results[0].boxes.conf.cpu()
    bbox_ids = results[0].boxes.id
    if bbox_ids is None and conf.shape[0] != 0: # If no tracking id, it returns -1
        bbox_ids =  torch.full((conf.shape[0],), -1) 
        print('Detections without tracking id, setting unknown')
    elif bbox_ids is None and conf.shape[0] == 0: # If no face is found it returns an empty tensor
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


def bbox_xywh2xyxy(bbox:torch.Tensor) -> torch.Tensor:
    """Converts the bbox from [x, y, w, h] format to [x1, y1, x2, y2] format.
    Params:
        - bbox (torch.Tensor): Bounding boxes in the format [x, y, w, h]. Expects the coordinates of the top-left corner 
            and its width and height. The shape is [n, 4], where n is the number of bboxes.
    Returns:
        - bbox (torch.Tensor): Bounding boxes in the format [x1, y1, x2, y2]. The shape is [n, 4], where n is the number of bboxes.
    """
    if bbox.shape[0] == 0:
        print("Empty bbox given")
        return bbox
    
    bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
    bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    return bbox



def bbox_xywh_center2xyxy(bbox:torch.Tensor) -> torch.Tensor:
    """Converts the bbox from [x, y, w, h] format where x,y represent the center, to [x1, y1, x2, y2] format.
    Params:
        - bbox (torch.Tensor): Bounding boxes in the format [x, y, w, h]. Expects the coordinates of the center 
            and its width and height (from top, left). The shape is [n, 4], where n is the number of bboxes.
    Returns:
        - bbox (torch.Tensor): Bounding boxes in the format [x1, y1, x2, y2] (left up point, right down point). 
        The shape is [n, 4], where n is the number of bboxes.
    """
    if bbox.shape[0] == 0:
        print("Empty bbox given")
        return bbox
    
    bbox[:, 0] = bbox[:, 0] - bbox[:, 2]/2
    bbox[:, 1] = bbox[:, 1] - bbox[:, 3]/2
    bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
    bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    return bbox



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
                face_threshold:float = 0.65, tracking:bool = False, get_object_confidence: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function to infer an image or frame using the given emotion model, face detector and hyperparameters. It returns 
    the faces detected with the bbox represented in xywh-center format, the predictions of the model (logits) 
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
        - get_object_confidence (bool): If True, it returns the confidence of the face detector. If True, it will not filter the faces with 
    Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The faces detected in xywh-center format, the predictions of the model 
            (logits) and the ids of the faces.
    """
    if tracking:
        [faces_bbox, bbox_ids, confidence_YOLO] = track_faces_YOLO(img, face_model, device = device, format = 'xywh-center')
    else:
        [faces_bbox, confidence_YOLO] = detect_faces_YOLO(img, face_model, format = 'xywh-center')
        bbox_ids = torch.arange(len(faces_bbox), dtype=torch.int)
    
    filtered_faces_bbox = faces_bbox[confidence_YOLO > face_threshold]
    filtered_ids = bbox_ids[confidence_YOLO > face_threshold]
    
    if len(filtered_faces_bbox) != 0:
        img_height, img_width, _ = img.shape
        squared_bbox = transform_bbox_to_square(filtered_faces_bbox, img_width, img_height)
        face_batch = create_faces_batch(img, face_transforms, squared_bbox, device)
        with torch.no_grad():
            face_batch.to(device)
            if distilled_model:
                raw_preds = arch.get_pred_distilled_model(emotion_model, face_batch, distilled_embedding_method)
            else:
                raw_preds = emotion_model(face_batch) # Predict emotions returns a tensor with logits
    else: # If no faces are detected, return empty tensors
        raw_preds = torch.empty(0).to(device)

    if get_object_confidence:
        return filtered_faces_bbox, raw_preds, filtered_ids, confidence_YOLO
    else:
        return filtered_faces_bbox, raw_preds, filtered_ids


def init_people_tracked(device:torch.device, window_size:int, preallocated_ids:int = 100000) -> torch.Tensor:
    """Function to initialize the people tracked as a torch.Tensor. Each row represents a person 
    and each column the N'th last emotion prediction.
    Params:
        - device (torch.device): The device to be used.
        - window_size (int): The size of the window to save the last window_size 
        emotions (including the current) for the people detected.
        - preallocated_ids (int): The number of ids to be preallocated. It is used 
        to avoid resizing the tensor each time a new detections appears.
    Returns:
        - torch.Tensor: The people tracked tensor. It has shape [preallocated_ids, 
        window_size, NUMBER_OF_EMOT], where NUMBER_OF_EMOT is the number of emotions. The first row 
        is used to store an special id for unknown tracking. So the track ids should start from 1.
    """
    people_tracked = torch.zeros((preallocated_ids, window_size, NUMBER_OF_EMOT)).to(device)
    people_tracked[0] = torch.ones(window_size,NUMBER_OF_EMOT).to(device) # If no tracking id, it returns a uniform distribution, it is located at the start of the tensor
    return people_tracked

    

def postprocessing_inference(people_tracked:torch.Tensor, preds:torch.Tensor, bbox_ids:torch.Tensor, 
                             device:torch.device, saving_prediction:str, mode:str = 'standard') -> Tuple[dict, torch.Tensor]:
    """Function to update the people_tracked with the new predictions and return the output predictions based 
    on the post processing mode. Depending on the saving_prediction parameter, it will save the predictions in logits 
    or in normalized version.
    Args:
        - people_tracked (torch.Tensor): The updated people_tracked. It has the bbox_id as row and per each column the 
        last N predictions of the model. The first row is used to store an special id for unknown tracking.
        - preds (torch.Tensor): The predictions of the model. It has shape [D, NUMBER_OF_EMOT], where D are the detections. 
            It is in logits
        - bbox_ids (torch.Tensor): The ids of the faces. It has shape [D].
        - device (torch.device): The device to be used.
        - saving_prediction (str): The method to save the predictions. It can be 'logits' or 'distrib'.
        - mode (str): The mode to be used for the postprocessing. It can be 'standard' or 'temporal_average'.
    Returns:
        - torch.Tensor: The updated people_tracked. It has the bbox_id as row and per each column the 
        last N predictions of the model. The first row is used to store an special id for unknown tracking.
        - torch.Tensor: The updated labels list. It has shape [D, NUMBER_OF_EMOT], where D are the detections. 
        - torch.Tensor: The sum of the predictions along the dimension 1. It has shape [D], where D are the detections.
    """
    if bbox_ids.shape[0] == 0:
        return people_tracked, torch.zeros(1,NUMBER_OF_EMOT).to(device), None
    
    assert saving_prediction in ['logits', 'distrib'], f"Invalid saving_prediction given: {saving_prediction}"

    if saving_prediction == 'logits':
        pass
    elif saving_prediction == 'distrib':
        preds = arch.get_distributions(preds) # Normalize the output to ensure it is 0-1
    

    if mode == 'standard':
        return people_tracked, preds, torch.ones(preds.shape[0]).to(device)
    
    elif mode == 'temporal_average':
        processed_preds = torch.zeros((preds.shape[0], NUMBER_OF_EMOT)).to(device)
        # Update the people tracked based on bbox ids and new pred, delete the last element of the tensor. It is added 1 to bbox_ids as 0 is special id for unknown tracking
        people_tracked[bbox_ids] = torch.cat((preds.unsqueeze(1), people_tracked[bbox_ids, :-1]), dim = 1)
        mean_preds = torch.mean(people_tracked[bbox_ids], dim = 1) # Compute mean accross each output logit
        if saving_prediction == 'logits':
            processed_preds = mean_preds
        elif saving_prediction == 'distrib':
            processed_preds = torch.divide(mean_preds, torch.sum(mean_preds, dim=1, keepdim=True))  # Normalize the output to ensure it is 0-1 (as in init there may be norm preds with 0 init vectors)
        return people_tracked, processed_preds, torch.sum(mean_preds, dim=1)
    else:
        raise ValueError(f"Invalid mode given for postprocessing inference: {mode}")
    


def standard_inference(ids, raw_preds, device, params):
    if params['postprocessing'] != 'standard': # If tracking is disabled, only standard postprocessing is allowed
        raise ValueError(f"Invalid postprocessing mode given: {params['postprocessing']} for tracking disabled. Only standard mode is allowed.")
    if ids.shape[0] == 0: # If no faces are detected, return empty tensors
        processed_preds = torch.zeros(1,NUMBER_OF_EMOT).to(device)
    else:
        processed_preds = raw_preds
        if params['saving_prediction'] == 'distrib':
            processed_preds = arch.get_distributions(processed_preds)
    return processed_preds



def set_confidence_threshold(processed_preds:torch.Tensor, sum_preds:torch.Tensor, emotion_threshold:float, device:torch.device) -> torch.Tensor:
    """Function to set the confidence threshold to the predictions. If the sum of the predictions is not one, it means the N
    previous predictions were not full, so it is set to -1. If the maximum value of the output_preds is below the confidence
    threshold, it is set to -1.
    Params:
        - processed_preds (torch.Tensor): The processed predictions. It has shape [D, NUMBER_OF_EMOT], where D are the detections.
        - sum_preds (torch.Tensor): The sum of the predictions along the dimension 1. It has shape [D], where D are the detections.
        - emotion_threshold (float): The confidence threshold to be used.
        - device (torch.device): The device to be used.
    Returns:
        - torch.Tensor: The processed predictions with the confidence threshold applied. It has shape [D, NUMBER_OF_EMOT], where D are the detections.
    """
    sum_not_one = (1 - sum_preds) > 1e-3 # Check if the sum is not one so the N previous predictions were not full
    processed_preds[sum_not_one] =  torch.zeros(NUMBER_OF_EMOT).to(device) # If the sum is not one, it means the N previous predictions were not full, so it is set to -1
    # Get the maximum value of output_preds along dimension 1
    max_values = torch.max(processed_preds, dim=1)[0]
    # Check if the maximum value is above the confidence threshold
    under_threshold = max_values < emotion_threshold
    # If the maximum value is not above the confidence threshold, set output_preds to -1
    processed_preds[under_threshold] = torch.zeros(NUMBER_OF_EMOT).to(device)
    return processed_preds
    



def get_pred_from_frame(frame:np.array, face_model:ultralytics.YOLO, emotion_model:torch.nn.Module, device:torch.device, 
                         face_transforms:albumentations.Compose, people_tracked: dict, params:dict, get_confidences:bool = False
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Function to get the predictions from a frame using the given models and hyperparameters. It returns the faces detected
    with the bbox represented in xywh-center, the labels of the model and the ids of the faces.
    Params:
        - frame (np.array): The frame to be inferred.
        - face_model (ultralytics.YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - device (torch.device): The device to be used.
        - face_transforms (albumentations.Compose): The face transformations.
        - people_tracked (dict): The dictionary with the people being tracked. It has as key the id of the face and value a list 
            of the last N outputs of the FER model, depending on params it will contain the logits or the normalized distribution.
        - params (dict): The dictionary with the hyperparameters.
        - get_object_confidence (bool): If True, it returns the confidence of the face detector.
    Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]: The faces detected, the labels of the model and the 
        ids of the faces and the updated list of people detected.
    """
    if get_confidences:
        faces_bbox, raw_preds, ids, object_confidence = get_raw_pred_from_frame(frame, face_model, emotion_model, params['distilled_model'], params['distilled_model_out_method'], 
                                                 device, face_transforms, params['face_threshold'], params['tracking'], get_confidences)
    else:
        faces_bbox, raw_preds, ids = get_raw_pred_from_frame(frame, face_model, emotion_model, params['distilled_model'], params['distilled_model_out_method'], 
                                                 device, face_transforms, params['face_threshold'], params['tracking'])
        
    sum_preds = torch.ones(ids.shape[0]).to(device) # Initialize the sum of the predictions to 1
    if params['tracking']:
        people_tracked, processed_preds, sum_preds = postprocessing_inference(people_tracked, raw_preds, ids, device,
                                                                    params['saving_prediction'], params['postprocessing'])
    else: 
        processed_preds = standard_inference(ids, raw_preds, device, params)
        
    if params['confident_emotion_prediction'] and params['saving_prediction'] == 'distrib':
        if ids.shape[0] > 0: # If no faces are detected, return empty tensors
            processed_preds = set_confidence_threshold(processed_preds, sum_preds, params['emotion_threshold'], device)
    else:
        assert params['confident_emotion_prediction'] == False, "Confident emotion prediction is only allowed when saving_prediction is 'distrib'."

                                                                    
    if params['saving_prediction'] == 'logits':
        labels = arch.get_predictions(processed_preds)
    elif params['saving_prediction'] == 'distrib':
        labels = arch.get_predictions_distrib(processed_preds)

    if get_confidences:
        return faces_bbox, labels, ids, processed_preds, people_tracked, object_confidence
    else:

        return faces_bbox, labels, ids, processed_preds, people_tracked