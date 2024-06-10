import os
import argparse
import pandas as pd
from tqdm import tqdm
import time

import torch
import cv2
import numpy as np 
import wandb

from src import TEST_VIDEO_DIR
from src.models import architectures as arch
from src.models import architectures_video as arch_v
from src.models.POSTER_V2.main import *
from src.models import architectures_video as arch_v

from config import wandbAPIkey



def main(size:str = None) -> None:
    """ Main function to evaluate the model. If a wandb_id is provided, the model weights are downloaded from 
    the Weights and Biases server. If not, the parameters are read from the params.yaml file. The model is then
    loaded and the test set is evaluated. If distillation is enabled, the model is evaluated using the three
    different embedding methods. The results are logged to the Weights and Biases server.
    Params:
        - wandb_id (str): The id of the Weights and Biases run to download the model weights.
    Returns:
        - None
    """
    # Load the emotion model
    wandb.login(key=wandbAPIkey)
    api = wandb.Api()
    artifact_dir = arch.get_wandb_artifact("sage-sweep-9", api = api)
    local_artifact = torch.load(os.path.join(artifact_dir, "model_best.pt"))
    params = local_artifact["params"]
    face_model, _, _, _, device = arch_v.load_video_models("sage-sweep-9", size, False, False)
    face_model_cpu, _, _, _, device_cpu = arch_v.load_video_models("sage-sweep-9", size, False, True)
    # List all files in the video directory and obtain the annotations
    annotations_path = os.path.join(TEST_VIDEO_DIR, 'video_annotations.pkl')
    video_dir = os.path.join(TEST_VIDEO_DIR, 'videos')
    video_files = os.listdir(video_dir)
    video_files = [f for f in video_files if f.endswith(('.mp4', '.avi', '.mov'))]  # Filter out non-video files
    total_frames = 0
    timings = np.array([])
    timings_cpu = np.array([])

    # Iterate over each video file
    for video_file in video_files:
        if video_file == 'no_audio_gopro_recording1.mp4':
            cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            new_timings = np.zeros(num_frames)
            new_timings_cpu = np.zeros(num_frames)
            for frame_id in tqdm(range(num_frames)): # All the frames in the video have a row in the df annotations, even if no bbox is present
                # Capture video frame-by-frame
                ret, frame = cap.read()
                torch.cuda.synchronize()
                start = time.time()
                _ = arch_v.detect_faces_YOLO(frame, face_model, format = 'xywh-center', verbose = False, device = device)
                end = time.time()
                new_timings[frame_id] = end - start
                torch.cuda.synchronize()
                start_cpu = time.time()
                _ = arch_v.detect_faces_YOLO(frame, face_model_cpu, format = 'xywh-center', verbose = False, device = device_cpu)
                end_cpu = time.time()
                new_timings_cpu[frame_id] = end_cpu - start_cpu

            
            total_frames += num_frames
            timings = np.concatenate((timings, new_timings))
            timings_cpu = np.concatenate((timings_cpu, new_timings_cpu))
            
            cap.release()
    mean_latency = np.sum(timings) / total_frames
    print(f"Mean latency: {mean_latency*1000:.2f} ms")
    mean_latency_cpu = np.sum(timings_cpu) / num_frames
    print(f"Mean latency CPU: {mean_latency_cpu*1000:.2f} ms")

def parse_args():
    """Parse the arguments of the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=str, default=None, help='Size of face detector model')
    return parser.parse_args() 


if __name__ == '__main__':
    args = parse_args()
    main(args.size)