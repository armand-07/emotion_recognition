import requests
import shutil
import bz2
import os

from src import MODELS_DIR


def download_MMOD_CNN_pretrained():
    """ Downloads the pretrained model for the MMOD CNN face detector.
    """
    # URL of the file to be downloaded
    url = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
    # Send a HTTP request to the URL of the file
    response = requests.get(url, stream=True)

    # Decompress the response content directly into a file
    decompressor = bz2.BZ2Decompressor()
    with open(os.path.join(MODELS_DIR, "mmod_human_face_detector.dat"), 'wb') as f_out:
        for data in response.iter_content(chunk_size=1024):
            f_out.write(decompressor.decompress(data))


