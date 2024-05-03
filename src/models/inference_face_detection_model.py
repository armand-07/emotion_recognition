from typing import Tuple

import cv2
import torch
import ultralytics
import numpy as np 

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