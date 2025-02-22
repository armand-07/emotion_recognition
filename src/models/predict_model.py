import argparse
import os
import time
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
import yaml


import cv2
from pyvirtualcam import Camera, PixelFormat
import matplotlib.colors as mcolors
try:
    import pyautogui as pg
except: # For the case of running in a server
    pass
import torch
import numpy as np
import albumentations
import ultralytics

from src import INFERENCE_DIR, NUMBER_OF_EMOT, EMOT_COLORS
import src.models.architectures_video as arch_v
from src.visualization.display_annot import plot_bbox_emot, plot_mean_emotion_distribution, plot_mean_emotion_evolution, create_figure_mean_emotion_distribution, create_figure_mean_emotion_evolution



def infer_screen(face_model:ultralytics.YOLO, emotion_model: torch.nn.Module, device: torch.device, 
                face_transforms:albumentations.Compose, EMOT_COLORS_RGB:list, params:dict) -> None:
    """Function to make inference in a screen. It shows the results in a window.
    Args:
        - face_model (ultralytics.YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - device (torch.device): The device to be used.
        - face_transforms (albumentations.Compose): The face transforms.
        - EMOT_COLORS_RGB (list): The list of RGB colors for the emotions in the same order as the emotions.
        - params (dict): The parameters to be used for the inference.
    Returns:
        - None
    """
    # Get the video properties
    screenshot = np.array(pg.screenshot())
    height, width, _ = screenshot.shape

    # Check if virtual camera should be used
    use_virtual_cam = params['vcam']

    if use_virtual_cam:
        # Initialize virtual camera
        virtual_cam = Camera(width=width, height=height, fps=fps_camera, fmt=PixelFormat.BGR)
        print("Using virtual camera")
    else:
        # Create OpenCV window
        cv2.namedWindow('Streaming inference', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Streaming inference', width, height)
    # Set variables if parameters are activated
    if params['tracking']:
        people_tracked = arch_v.init_people_tracked(device, params['window_size']) # Initialize the emotion tracker
    else:
        people_tracked = None
    if params['show_mean_emotion_distrib']:
        fig_distrib, ax_distrib, distribution_container = create_figure_mean_emotion_distribution(height, width)
    if params['show_mean_emotion_evolution']:
        fig_evolution, ax_evolution, evolution_container = create_figure_mean_emotion_evolution(height, width)
        last_mean_emotions = torch.zeros(NUMBER_OF_EMOT, params['evolution_frames']).to(device)

    # Read until video is completed
    while True:
        frame = np.array(pg.screenshot())
        faces_bbox, labels, ids, processed_preds, people_tracked = arch_v.get_pred_from_frame(frame, face_model, emotion_model, device, face_transforms, people_tracked, params)
        if params['view_emotion_model_attention'] and len(ids) != 0:
            cls_weight = emotion_model.base_model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(-1, 14, 14).detach().to('cpu')
        else:
            cls_weight = None
        frame = plot_bbox_emot(frame, faces_bbox, labels, ids, cls_weight, bbox_format ="xywh-center", display = False, color_list = EMOT_COLORS_RGB)
        # Display the mean sentiment of the people in the frame
        if params['show_mean_emotion_distrib']:
                frame, fig_distrib, ax_distrib, distribution_container = plot_mean_emotion_distribution(frame, processed_preds, params['saving_prediction'],
                                                                                                        fig_distrib, ax_distrib, distribution_container)
        if params['show_mean_emotion_evolution']: 
                frame, last_mean_emotions, fig_evolution, ax_evolution, evolution_container = plot_mean_emotion_evolution(frame, processed_preds, last_mean_emotions, params['saving_prediction'], 
                                                                                                                EMOT_COLORS_RGB, fig_evolution, ax_evolution, evolution_container)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert the frame to standard BGR before printing
        cv2.imshow('Streaming inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press Q on keyboard to exit
            break


def infer_stream(cap:cv2.VideoCapture, face_model:ultralytics.YOLO, emotion_model: torch.nn.Module, device: torch.device, 
                face_transforms:albumentations.Compose, EMOT_COLORS_RGB:list, params:dict) -> None:
    """Function to make inference in a video stream. It shows the results in a window.
    Args:
        - cap (cv2.VideoCapture): The video capture object.
        - face_model (ultralytics.YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - device (torch.device): The device to be used.
        - face_transforms (albumentations.Compose): The face transforms.
        - EMOT_COLORS_RGB (list): The list of RGB colors for the emotions in the same order as the emotions.
        - params (dict): The parameters to be used for the inference.
    Returns:
        - None
    """
    # Get the video properties
    fps_camera = cap.get(cv2.CAP_PROP_FPS)  
    print(f"Camera FPS: {fps_camera}")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Camera resolution: {width}x{height}")

    # Check if virtual camera should be used
    use_virtual_cam = params["vcam"]
    if use_virtual_cam:
        virtual_cam = Camera(width=width, height=height, fps=fps_camera, fmt=PixelFormat.BGR)
    else:
        cv2.namedWindow('Streaming inference', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Streaming inference', width, height)

    # Set variables if parameters are activated
    people_tracked = None
    if params['tracking']:
        people_tracked = arch_v.init_people_tracked(device, params['window_size']) # Initialize the emotion tracker
        
    if params['show_mean_emotion_distrib']:
        fig_distrib, ax_distrib, distribution_container = create_figure_mean_emotion_distribution(height, width)
    if params['show_mean_emotion_evolution']:
        fig_evolution, ax_evolution, evolution_container = create_figure_mean_emotion_evolution(height, width)
        last_mean_emotions = torch.zeros(NUMBER_OF_EMOT, params['evolution_frames']).to(device)

    # Read until video is completed
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB
            faces_bbox, labels, ids, processed_preds, people_tracked = arch_v.get_pred_from_frame(frame, face_model, emotion_model, 
                                                                                                   device, face_transforms, people_tracked, params)
            if params['view_emotion_model_attention'] and len(ids) != 0: # If there are faces detected it can be observed the attention map
                cls_weight = emotion_model.base_model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(-1, 14, 14).detach().to('cpu')
            else:
                cls_weight = None
            frame = plot_bbox_emot(frame, faces_bbox, labels, ids, cls_weight, bbox_format ="xywh-center", display = False, color_list = EMOT_COLORS_RGB)
            # Display the mean sentiment of the people in the frame
            if params['show_mean_emotion_distrib']:
                frame, fig_distrib, ax_distrib, distribution_container = plot_mean_emotion_distribution(frame, processed_preds, params['saving_prediction'],
                                                                                                        fig_distrib, ax_distrib, distribution_container)
            if params['show_mean_emotion_evolution']: 
                frame, last_mean_emotions, fig_evolution, ax_evolution, evolution_container = plot_mean_emotion_evolution(frame, processed_preds, last_mean_emotions, params['saving_prediction'], 
                                                                                                                EMOT_COLORS_RGB, fig_evolution, ax_evolution, evolution_container)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert the frame to standard BGR before printing
            
            if use_virtual_cam:
                # Send frame to virtual camera
                virtual_cam.send(frame)
                virtual_cam.sleep_until_next_frame()
            else:
                # Show frame in OpenCV window
                cv2.imshow('Streaming inference', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Presionar Q en el teclado para salir
                    break

        else:
            break
    # Release the capture and close the window
    cap.release()
    if not use_virtual_cam:
        cv2.destroyAllWindows()


def infer_video_and_save(cap: cv2.VideoCapture, output_cap: cv2.VideoWriter, name:str, face_model: ultralytics.YOLO, 
                        emotion_model: torch.nn.Module, device: torch.device, face_transforms: albumentations.Compose,
                        EMOT_COLORS_RGB:list, params:dict) -> None:
    """Function to make inference in a video and save the result in capturer.
    Args:
        - cap (cv2.VideoCapture): The video capture object.
        - output_cap (cv2.VideoWriter): The video writer object.
        - name (str): The name of the video.
        - face_model (YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - device (torch.device): The device to be used.
        - face_transforms (albumentations.Compose): The face transforms.
        - EMOT_COLORS_RGB (list): The list of RGB colors for the emotions in the same order as the emotions.
        - params (dict): The parameters to be used for the inference.
    Returns:
        - None
    """
    # Get the video properties
    fps_camera = cap.get(cv2.CAP_PROP_FPS)  
    print(f"Camera FPS: {fps_camera}")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Camera resolution: {width}x{height}")
    # Set variables if parameters are activated
    if params['tracking']:
        people_tracked = arch_v.init_people_tracked(device, params['window_size']) # Initialize the emotion tracker
    else:
        people_tracked = None
    if params['show_mean_emotion_distrib']:
        fig_distrib, ax_distrib, distribution_container = create_figure_mean_emotion_distribution(height, width)
    if params['show_mean_emotion_evolution']:
        fig_evolution, ax_evolution, evolution_container = create_figure_mean_emotion_evolution(height, width)
        last_mean_emotions = torch.zeros(NUMBER_OF_EMOT, params['evolution_frames']).to(device)
    if params['show_inference']:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, width, height)

    # Read until video is completed
    for frame_id in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc = f'Analizing video "{name}"'):
        # Capture frame-by-frame
        ret, frame = cap.read()     # ret is a boolean that returns True if the frame is available. frame is the image array.
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_bbox, labels, ids, processed_preds, people_tracked = arch_v.get_pred_from_frame(frame, face_model, emotion_model, device, 
                                                                                                face_transforms, people_tracked, params)
            if params['save_result'] or params['show_inference']: # Show the visual results if needed
                if params['view_emotion_model_attention'] and len(ids) != 0:
                    cls_weight = emotion_model.base_model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(-1, 14, 14).detach().to('cpu') 
                else:
                    cls_weight = None
                frame = plot_bbox_emot(frame, faces_bbox, labels, ids, cls_weight, bbox_format ="xywh-center", display = False, 
                                       color_list = EMOT_COLORS_RGB)
                # Display the mean sentiment of the people in the frame
                if params['show_mean_emotion_distrib']:
                    frame, fig_distrib, ax_distrib, distribution_container = plot_mean_emotion_distribution(frame, processed_preds, params['saving_prediction'],
                                                                                                            fig_distrib, ax_distrib, distribution_container)
                if params['show_mean_emotion_evolution']: 
                    frame, last_mean_emotions, fig_evolution, ax_evolution, evolution_container = plot_mean_emotion_evolution(frame, processed_preds, last_mean_emotions, params['saving_prediction'], 
                                                                                                                EMOT_COLORS_RGB, fig_evolution, ax_evolution, evolution_container)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert the frame to standard BGR before printing or saving
                if params['show_inference']:
                    cv2.imshow(name, frame)
                    cv2.waitKey(1)      # Wait 1ms to be able to see properly the results
                if params['save_result']: # Write the frame into the output file
                    output_cap.write(frame)

        else: # Break the loop if video has ended
            break



def process_file(input_path:str, output_dir:str, face_model: ultralytics.YOLO, emotion_model:torch.nn.Module, device: torch.device, 
                 face_transforms: albumentations.Compose, EMOT_COLORS_RGB:list,  params: dict) -> None:
    """Function to process a file, either an image or a video. It saves the results in the output_dir, and if needed shows realtime 
    results.
    Args:
        - input_path (str): The path to the file to be processed.
        - output_dir (str): The directory to save the results.
        - face_model (YOLO): The face detector model.
        - emotion_model (torch.nn.Module): The emotion model.
        - device (torch.device): The device to be used.
        - face_transforms (albumentations.Compose): The face transforms.
        - EMOT_COLORS_RGB (list): The list of RGB colors for the emotions in the same order as the emotions.
        - params (dict): The parameters to be used for the inference.
    Returns:
        - None 
    """
    if input_path.endswith('.jpg') or input_path.endswith('.png'):
        # Set tracking off for images
        params_image = dict(params)
        params_image['tracking'] = False
        people_tracked = None
        params_image['postprocessing'] = 'standard'

        # Read image
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if params['show_mean_emotion_distrib']:
            img_height, img_width, _ = img.shape
            fig, ax, distribution_container = create_figure_mean_emotion_distribution(img_height, img_width)

        # Make inference
        start = time.time()
        faces_bbox, labels, ids, processed_preds, people_tracked = arch_v.get_pred_from_frame(img, face_model, emotion_model, 
                                                                                              device, face_transforms, 
                                                                                              people_tracked, params_image)
        if params['save_result'] or params['show_inference']: # Show the visual results if needed
            if params['view_emotion_model_attention'] and len(ids) != 0:
                cls_weight = emotion_model.base_model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(-1, 14, 14).detach().to('cpu')
            else:
                cls_weight = None
            img = plot_bbox_emot(img, faces_bbox, labels, ids, cls_weight, bbox_format ="xywh-center", display = False, 
                                 color_list = EMOT_COLORS_RGB)
            if params['show_mean_emotion_distrib']:
                img, fig, ax, distribution_container = plot_mean_emotion_distribution(img, processed_preds, fig, ax, 
                                                                                      distribution_container)
        end = time.time()
        print(f"The time needed to process the image: {end - start:.2f}s")

        # Display the results
        if params['show_inference']:
            cv2.imshow('Image', img)
        if params['save_result']:
            # Define the output path
            name = os.path.basename(input_path).split('.')[0]
            output_filename = os.path.join(output_dir, name+"_inference.jpg")
            if os.path.exists(output_filename):
                os.remove(output_filename)
            # Save the image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR to proper saving
            cv2.imwrite(output_filename, img)
            print("Saving in:", output_filename)


    elif input_path.endswith('.mp4') or input_path.endswith('.avi') or input_path.endswith('.mov'):
        cap = cv2.VideoCapture(input_path)
        # Get the width, height, total frames and fps of the video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the output path
        name = os.path.basename(input_path).split('.')[0]
        output_filename = os.path.join(output_dir, name+"_inference.mp4")
        if os.path.exists(output_filename):
            os.remove(output_filename)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_cap = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
        # Make inference
        start = time.time()
        infer_video_and_save(cap, output_cap, name, face_model, emotion_model, device, face_transforms, EMOT_COLORS_RGB, params)
        end = time.time()
        # Store results and release the video capture object
        cap.release()
        output_cap.release()
        print(f"The time needed to process the video: {end - start:.2f}s")
        print(f"The average time per frame: {(end - start)/total_frames:.5f}s")
        print(f"It is {((end - start)/total_frames)/(1/fps):.2f} times slower than real time inference with the video's {fps} fps")
        print("The video has been saved in:", output_filename)
    
    else:
        raise ValueError(f"Invalid archive type given: {os.path.basename(input_path).split('.')[1]}")



def main(mode: str, input_path: str, output_dir:str, cpu:bool, camera_id:int, vcam:bool) -> None:
    """Main function to run the inference of the model. It can make streaming inference, on a set of files or only a file.
    Args:
        - mode (str): The mode to be used for the inference.
        - input_path (str): The input file to be used for the inference.
        - output_dir (str): The directory to save the results.
        - cpu (bool): Perform inference on CPU
        - camera_id (int): The camera to be used for the streaming inference.
    Returns:
        - None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Path of the parameters file
    params_path = Path("params.yaml")
    # Read data preparation parameters
    with open(params_path, "r", encoding='utf-8') as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["inference"]
        except yaml.YAMLError as exc:
            print(exc)
    params['job_type'] = "test"

    face_model, emotion_model, distilled_model, face_transforms, device = arch_v.load_video_models(params['wandb_id_emotion_model'], params['face_detector_size'], 
                                                                                                   params['view_emotion_model_attention'], cpu)
    params['distilled_model'] = distilled_model
    if params['emotion_threshold'] > 0:
        params['confident_emotion_prediction'] = True # To avoid predictions when the number of frames in not full
    else:
        params['confident_emotion_prediction'] = False
    
    if params['variable_color']: # If the color of the emotions is variable, it needs to be converted to RGB
        # Convert color names to RGB values
        EMOT_COLORS_RGB = [mcolors.to_rgb(color) for color in EMOT_COLORS]
        # Convert RGB values to the range of 0 to 255
        EMOT_COLORS_RGB = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in EMOT_COLORS_RGB]
    else:
        EMOT_COLORS_RGB = None

    # Start with inference
    if mode == 'stream':
        cap = cv2.VideoCapture(camera_id)
        params['vcam'] = vcam
        infer_stream(cap, face_model, emotion_model, device, face_transforms, EMOT_COLORS_RGB, params)

    elif mode == 'screen':
        params['vcam'] = vcam
        infer_screen(face_model, emotion_model, device, face_transforms, EMOT_COLORS_RGB, params)

    elif mode == 'save':
        input_path = os.path.join(INFERENCE_DIR, input_path)
        if os.path.isdir(input_path):
            print(f"Found a directory")
            files = os.listdir(input_path)
            first_execution = True
            for file in files:
                input_file_path = os.path.join(input_path, file)
                if params['tracking'] and not first_execution: # If tracking is enabled, it needs to restart the tracking by realoading the model
                    face_model = arch_v.load_YOLO_model_face_recognition(size = params['face_detector_size'], device = device)
                process_file(input_file_path, output_dir, face_model, emotion_model, device, face_transforms, EMOT_COLORS_RGB, params)
                first_execution = False
        elif os.path.isfile(input_path):
            print(f"Found an archive")
            process_file(input_path, output_dir, face_model, emotion_model, device, face_transforms, EMOT_COLORS_RGB, params)
        else:
            print(f"{input_path} is neither a directory nor an archive file")

    else:
        raise ValueError(f"Invalid mode given: {mode}")
    


def parse_args():
    """Function to parse the arguments of the command line. It returns the arguments as a Namespace object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='save', help='Process to stream results from camera, or make inference to saved img or video')
    parser.add_argument('--input_path', type=str, default = 'test', help= 'The file to be used for the inference. If mode is cam, it is ignored. If mode is video or img, it is the path to the archive.')
    parser.add_argument('--output_dir', type=str, default = os.path.join(INFERENCE_DIR, 'output'), help= 'Directory to save results')
    parser.add_argument('--cpu', action=argparse.BooleanOptionalAction, help= 'Perform all the inference on CPU on non GPU hardware')
    parser.add_argument('--camera', type=int, default=0, help= 'The camera to be used for the streaming inference')
    parser.add_argument('--vcam', action=argparse.BooleanOptionalAction, help= 'Create virtual camera to show the results')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    main(args.mode, args.input_path, args.output_dir, args.cpu, args.camera, args.vcam)