import cv2
from torchvision import transforms
from PIL import Image
import os
import dlib
from torch_mtcnn import detect_faces

from src import MODELS_DIR

def detect_faces_HAAR_cascade(img, pretrained_model):
    """ Detects faces in an image using the given Haar cascade pretrained model. 
    Returns in standard bbox format: [x, y, w, h]
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Perform face detection
    faces_bbox = pretrained_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    return faces_bbox

def detect_faces_HOG_SVM(img, pretrained_model):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB channel ordering (which is what dlib expects)
    det = pretrained_model(rgb, 1) # Number of times to upsample an image before applying face detection
    faces_bbox = []
    for i, d in enumerate(det):
            faces_bbox.append([d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top()])
    return faces_bbox