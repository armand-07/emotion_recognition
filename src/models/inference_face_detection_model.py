import cv2
import numpy as np

def detect_faces_HAAR_cascade(img, pretrained_model):
    """ Detects faces in an image using the given Haar cascade pretrained model. 
    Returns in standard bbox format: [x, y, w, h]
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform face detection
    faces_bbox = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    return faces_bbox


def transform_bbox_to_square(bboxes, img_width, img_height):
    """Expects the coordinates of the center of the bbox and its width and height. Then returns the 
    coordinates of the top-left corner of the bbox and the width and height of the bbox, so that it is a square. 
    It also ensures the result is inside the image"""
    max_dims = np.maximum(bboxes[:, 2], bboxes[:, 3])  # Get the maximum between width and height per each detected face
    bboxes[:, 2] = max_dims  # Update width
    bboxes[:, 3] = max_dims  # Update height
    bboxes[:, 0] = bboxes[:, 0] - max_dims / 2  # Update x coordinates to top-left corner
    bboxes[:, 1] = bboxes[:, 1] - max_dims / 2  # Update y coordinates to top-left corner
    # Ensure the bbox is inside the image
    bboxes[:, 0] = np.maximum(bboxes[:, 0], 0)
    bboxes[:, 1] = np.maximum(bboxes[:, 1], 0)
    bboxes[:, 2] = np.minimum(bboxes[:, 2], img_width - bboxes[:, 0])
    bboxes[:, 3] = np.minimum(bboxes[:, 3], img_height - bboxes[:, 1])
    return bboxes


def detect_faces_YOLO(img, pretrained_model, format = 'xywh', verbose=False, stay_on_gpu=False):
    """ Detects faces in an image using the given YOLO pretrained model. 
    Returns in standard bbox format: [x, y, w, h], where x,y are the coordinates of the top-left corner. And the condifece of the detection.
    """
    results = pretrained_model.predict(img, verbose = verbose)

    # Extract the bounding boxes and probabilities for the image
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
    
    conf = results[0].boxes.conf

    if not stay_on_gpu:
        boxes = boxes.cpu().numpy()
        conf = conf.cpu().numpy()
    

    return [boxes, conf]