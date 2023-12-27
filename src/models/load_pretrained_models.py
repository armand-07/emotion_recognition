import os
import cv2
from ultralytics import YOLO

from src import MODELS_DIR  
from src.models.download_pretrained_models import download_YOLO_model_face_recognition


def load_HAAR_cascade_face_detection():
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


def load_YOLO_model_face_recognition(device, size = "medium",  directory=MODELS_DIR):
    """ Downloads the pretrained model for the YOLO depending on specified size.
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
    model = YOLO(model_path)
    
    return model
        
