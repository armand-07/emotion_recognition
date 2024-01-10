import cv2

def detect_faces_HAAR_cascade(img, pretrained_model):
    """ Detects faces in an image using the given Haar cascade pretrained model. 
    Returns in standard bbox format: [x, y, w, h]
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform face detection
    faces_bbox = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    return faces_bbox

def detect_faces_YOLO(img, pretrained_model, format = 'xywh', verbose=False):
    """ Detects faces in an image using the given YOLO pretrained model. 
    Returns in standard bbox format: [x, y, w, h], where x,y are the coordinates of the top-left corner. And the condifece of the detection.
    """
    results = pretrained_model.predict(img, verbose = verbose)

    # Extract the bounding boxes and probabilities for the image
    if format == 'xywh-center':
        boxes = results[0].boxes.xywh.cpu().numpy()
    elif format == 'xywh':
        boxes = results[0].boxes.xywh.cpu().numpy()
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2]/2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
    elif format == 'xyxy':
        boxes = results[0].boxes.xyxy.cpu().numpy()
    else:
        raise ValueError('Format not supported')
    
    conf = results[0].boxes.conf.cpu().numpy()

    return [boxes, conf]