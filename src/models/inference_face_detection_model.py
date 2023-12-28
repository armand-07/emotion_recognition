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

def detect_faces_YOLO(img, pretrained_model):
    """ Detects faces in an image using the given YOLO pretrained model. 
    Returns in standard bbox format: [x, y, w, h], where x,y are the coordinates of the top-left corner.
    """
    results = pretrained_model(img)

    # Extract the bounding boxes and probabilities for the image
    boxes = results[0].boxes
    probs = results[0].probs

    return [boxes, probs]