from typing import Tuple
from pathlib import Path
import yaml
import os
import argparse
from tqdm import tqdm
import pandas as pd
import time

import torch
from torcheval.metrics import MulticlassAccuracy
import torchvision
import cv2
import ultralytics
import albumentations
import wandb

from src import TEST_VIDEO_DIR, NUMBER_OF_EMOT
from src.models import architectures as arch
from src.models.metrics import save_video_test_wandb_metrics
from src.models import architectures_video as arch_v
from src.models.POSTER_V2.main import *

from config import wandbAPIkey



def eval_video(annotations:str, cap:cv2.VideoCapture, name: str, face_model: ultralytics.YOLO, emotion_model: torch.nn.Module, 
               device: torch.device, face_transforms: albumentations.Compose, params:dict,
               criterion: torch.nn , acc1:MulticlassAccuracy, acc2:MulticlassAccuracy,
               ) -> Tuple[float, float, int, int, float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
                          MulticlassAccuracy, MulticlassAccuracy]:
    # Define metrics
    total_loss = 0.0
    total_GTs, total_object_detections, total_emotion_detections = 0, 0, 0 # total_detections should be less than GT_people 
    sum_inference_time, sum_inference_time_people = 0.0 , 0.0
    all_GT_labels, all_preds_labels = torch.empty(0, device = 'cpu', dtype = torch.long), torch.empty(0, device = 'cpu', dtype = torch.long)
    obj_pred_confidences, obj_TP_preds = torch.empty(0, device = 'cpu'), torch.empty(0, device = 'cpu', dtype = torch.int)
    
    IoU_threshold = float(params['IoU_threshold'])
    
    if params['tracking']:
        people_tracked = arch_v.init_people_tracked(device, params['window_size']) # Initialize the emotion tracker
    else:
        people_tracked = None
    # Iterate over each frame of the video
    for frame_id in tqdm(range(annotations.shape[0])): # All the frames in the video have a row in the df annotations, even if no bbox is present
        # Capture video frame-by-frame
        ret, frame = cap.read()
        GT_bboxes = annotations.iloc[frame_id]['bboxes']
        GT_labels = annotations.iloc[frame_id]['labels']
        if ret == True: # If the frame has not ended
            if GT_labels[0].item() != -1: # Check if it has a GT bbox assigned to the frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Get the predictions from the frame
                start = time.time()
                faces_bbox, labels, ids, processed_preds, people_tracked, object_confidence = arch_v.get_pred_from_frame(frame, face_model, emotion_model, device, 
                                                                                                    face_transforms, people_tracked, params, get_object_confidence = True)
                end = time.time()

                # Update metrics
                sum_inference_time += end - start
                total_GTs += GT_labels.size(0) # Update the total number of total GT available in frame
                obj_pred_confidences = torch.cat((obj_pred_confidences, object_confidence), dim = 0)
                people_detected_frame = len(ids)
                total_object_detections += people_detected_frame
                if people_detected_frame > 0: # If there are people detected
                    sum_inference_time_people += (end - start) / people_detected_frame # Divide the inference time by the number of people detected
                    faces_bbox = arch_v.bbox_xywh_center2xyxy(faces_bbox) # Convert the bbox from xywh to xyxy to compute IoU
                    max_iou, bbox_idx_max_iou = torch.max(torchvision.ops.box_iou(GT_bboxes, faces_bbox), dim = 1) # Get the max iou per each GT bbox
                    
                    # Filter out the predictions with IoU values below the IoU threshold
                    filter_IoU = max_iou > IoU_threshold
                    bbox_idx_max_iou = bbox_idx_max_iou[filter_IoU]
                    GT_labels = GT_labels[filter_IoU]

                    # Measure face detector performance
                    TP_detections = torch.zeros(people_detected_frame, dtype = torch.int)
                    TP_detections[bbox_idx_max_iou] = 1
                    obj_TP_preds = torch.cat((obj_TP_preds, TP_detections), dim = 0)

                    # Filter out the GT labels and face preds with the label 8 (unknown)
                    label_filter = GT_labels != 8 
                    GT_labels = GT_labels[label_filter]; bbox_idx_max_iou = bbox_idx_max_iou[label_filter]
                    emotion_detections = bbox_idx_max_iou.size(0)
                    total_emotion_detections += emotion_detections
                    if emotion_detections > 0: # If there are bbox_preds with GT label with label != 8 (that is the label "DNC" (Do Not Care))
                        processed_preds = processed_preds[bbox_idx_max_iou] # Get the processed predictions of the GT bboxes with IoU above the threshold
                        processed_preds = arch.get_distributions(processed_preds).cpu() # Get the distributions of the predictions
                        # Compute loss
                        total_loss += criterion(processed_preds, GT_labels).item()
                        # Compute the accuracy
                        acc1.update(processed_preds, GT_labels)
                        acc2.update(processed_preds, GT_labels)
                        # Concatenate the labels
                        all_GT_labels = torch.cat((all_GT_labels, GT_labels), dim = 0)
                        all_preds_labels = torch.cat((all_preds_labels, torch.argmax(processed_preds, dim = 1).long()), dim = 0)
                else: # If no detection is found (id and bbox list empty), penalize only the face detector increasing total of GT detected and no IoU
                    sum_inference_time_people += end - start
        else: # Break the loop if video has ended
            break

    return [total_loss, total_GTs, total_object_detections, total_emotion_detections,
            sum_inference_time, sum_inference_time_people, 
            all_GT_labels, all_preds_labels, obj_pred_confidences, obj_TP_preds, acc1, acc2]



def eval_model_on_videos(annotations_path:str, video_dir:str, params:dict, run: wandb.run = None) -> None:
    """Evaluate the model on the test video set. The annotations are read from the annotations_path and the videos
    are read from the video_dir. The model is loaded and the test set is evaluated. The metrics are logged to the
    Weights and Biases server.
    Params:
        - annotations_path (str): The path to the annotations file.
        - video_dir (str): The path to the directory containing the videos.
        - params (dict): The parameters to be used for the evaluation.
        - run (wandb.run): The Weights and Biases run to log the metrics.
    Returns:
        - None"""
    # Load model and parameters
    face_model, emotion_model, distilled_model, face_transforms, device = arch_v.load_video_models(params['wandb_id_emotion_model'],
                                                                                                   params['face_detector_size'])
    params['distilled_model'] = distilled_model
    first_execution = True

    # Initialize metrics
    global_sum_loss = 0.0
    total_GTs, total_object_detections, total_emotion_detections = 0, 0, 0
    total_inference_time, total_inference_time_people = 0.0, 0.0
    total_frames = 0
    all_GT_labels, all_preds_labels = torch.empty(0, device = 'cpu', dtype = torch.long), torch.empty(0, device = 'cpu', dtype = torch.long)
    all_obj_pred_confidences, all_obj_TP_preds = torch.empty(0, device = 'cpu'), torch.empty(0, device = 'cpu', dtype = torch.int)
    acc1, acc2  = MulticlassAccuracy(device=device), MulticlassAccuracy(device=device, k = 2)
    criterion = torch.nn.CrossEntropyLoss(reduction = 'sum')
    
    # List all files in the video directory and obtain the annotations
    video_files = os.listdir(video_dir)
    df_annotations = pd.read_pickle(annotations_path)
    video_files = [f for f in video_files if f.endswith(('.mp4', '.avi', '.mov'))]  # Filter out non-video files

    # Iterate over each video file
    for video_file in video_files:
        video_filename = video_file.split(".")[0]
        if video_filename not in df_annotations['filename'].values:# or video_filename != 'izhar_recording2': # If the annotation file does not exist, skip this video
            print(f'Skipping video {video_filename} because no annotation was found in the processed annotations')
            continue
        else: # Analize the video
            print(f'Analizing video {video_filename}')
            if params['tracking'] and not first_execution: # If tracking is enabled, it needs to restart the tracking by realoading the face model
                face_model = arch_v.load_YOLO_model_face_recognition(size = params['face_detector_size'], device = device)
            df_video = df_annotations[df_annotations['filename'] == video_filename].set_index('frame')
            total_frames += df_video.shape[0] # Update the total number of frames based on the number of frames annotated in the video

            cap = cv2.VideoCapture(os.path.join(video_dir, video_file))

            [sum_loss_video, video_GTs, video_object_detections, video_emotion_detections,
             inference_time_video, inference_time_people_video,
             GT_labels_video, preds_labels_video, obj_pred_confidences, obj_TP_preds, 
             acc1, acc2] = eval_video(df_video, cap, video_filename, face_model, emotion_model, 
                                                                            device, face_transforms, params, criterion, acc1, acc2)
            first_execution = False # Set to False after the first execution to reload the face model on the next iteration
            
            # Update metrics
            global_sum_loss += sum_loss_video
            total_GTs += video_GTs
            total_object_detections += video_object_detections; total_emotion_detections += video_emotion_detections
            total_inference_time += inference_time_video; total_inference_time_people += inference_time_people_video
            all_GT_labels = torch.cat((all_GT_labels, GT_labels_video), dim = 0)
            all_preds_labels = torch.cat((all_preds_labels, preds_labels_video), dim = 0)
            all_obj_pred_confidences = torch.cat((all_obj_pred_confidences, obj_pred_confidences), dim = 0)
            all_obj_TP_preds = torch.cat((all_obj_TP_preds, obj_TP_preds), dim = 0)
            print(all_obj_pred_confidences.shape, all_obj_TP_preds.shape)
            print(all_obj_pred_confidences, all_obj_TP_preds)

    print(global_sum_loss, total_GTs, video_object_detections, total_emotion_detections, 
                                    total_inference_time, total_inference_time_people, total_frames,
                                    all_GT_labels, all_preds_labels)
            
    # Compute the metrics
    save_video_test_wandb_metrics(global_sum_loss, total_GTs, total_object_detections, total_emotion_detections,  
                                    total_inference_time, total_inference_time_people, total_frames,
                                    all_GT_labels, all_preds_labels, all_obj_pred_confidences, all_obj_TP_preds, acc1, acc2, run, params)



def main(wandb_id:str = None) -> None:
    """ Main function to evaluate the model. If a wandb_id is provided, the model weights are downloaded from 
    the Weights and Biases server. If not, the parameters are read from the params.yaml file. The model is then
    loaded and the test set is evaluated. If distillation is enabled, the model is evaluated using the three
    different embedding methods. The results are logged to the Weights and Biases server.
    Params:
        - wandb_id (str): The id of the Weights and Biases run to download the model weights.
    Returns:
        - None
    """
    wandb.login(key=wandbAPIkey)
    run = wandb.init(
    entity="armand-07",
    project="TFG FER video testing",
    job_type="video_test",
    )
    run_name = wandb.run.name
    print(f'WanDB run name is: {run_name}') # Print the run number hash id
    
    if wandb_id == None: # If no wandb_id is provided, use the params.yaml file
        # Path of the parameters file
        params_path = Path("params.yaml")
        # Read data preparation parameters
        with open(params_path, "r", encoding='utf-8') as params_file:
            try:
                params = yaml.safe_load(params_file)
                params = params["test_video"]
            except yaml.YAMLError as exc:
                print(exc)
    else: # If a wandb_id is provided, download the model weights
        artifact_dir = arch.get_wandb_artifact(wandb_id, run)
        local_artifact = torch.load(os.path.join(artifact_dir, "model_best.pt"))
        params = local_artifact["params"]
        print(f"Loaded best model at epoch {local_artifact['epoch']} from run {wandb_id}")
    wandb.config.update(params)

    if wandb_id == None: # If no wandb_id is provided, use the params.yaml file
        # Path of the parameters file
        params_path = Path("params.yaml")
        # Read data preparation parameters
        with open(params_path, "r", encoding='utf-8') as params_file:
            try:
                params = yaml.safe_load(params_file)
                params = params["test_video"]
            except yaml.YAMLError as exc:
                print(exc)
    
    annotations_path = os.path.join(TEST_VIDEO_DIR, 'video_annotations.pkl')
    video_dir = os.path.join(TEST_VIDEO_DIR, 'videos')
    # Evaluate the model on the test video set
    eval_model_on_videos (annotations_path, video_dir, params, run)
    
    wandb.finish()


def parse_args():
    """Parse the arguments of the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_id', type=str, default=None, help='Run id to take the model weights')
    return parser.parse_args() 


if __name__ == '__main__':
    args = parse_args()
    main(args.wandb_id)