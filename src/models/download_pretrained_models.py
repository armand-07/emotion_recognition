import requests
import os


from src import MODELS_DIR  


def download_YOLO_model_face_recognition(size="medium", directory=MODELS_DIR):
    """ Downloads the pretrained model for the YOLO depending on specified size.
    """
    # URL of the file to be downloaded
    # Replace with the URL of the YOLOv8 model
    if size == "nano":
        url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
    elif size == "medium":
        url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-face.pt"
    elif size == "large":
        url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt"
    
    # Send a HTTP request to the URL of the file
    filename = url.split("/")[-1]
    response = requests.get(url)

    model_path = os.path.join(directory, filename)
    with open(model_path, mode="wb") as file:
        file.write(response.content)


