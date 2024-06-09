from pathlib import Path
import os

# Define path variables
ROOT_DIR = Path(Path(__file__).resolve().parent.parent)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FACE_DETECT_DIR = os.path.join(MODELS_DIR, "face_recognition")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
INFERENCE_DIR = os.path.join(ROOT_DIR, "inference")

#AFFECTNET_DIR = os.path.normpath("/mnt/gpid08/datasets/affectnet/") # substitute with the actual data path
AFFECTNET_DIR = os.path.join(DATA_DIR, "affectnet")
RAW_AFFECTNET_DIR =  os.path.join(AFFECTNET_DIR, "raw") 
INTERIM_AFFECTNET_DIR = os.path.join(AFFECTNET_DIR, "interim")
PROCESSED_AFFECTNET_DIR = os.path.join(AFFECTNET_DIR, "processed")

#TEST_VIDEO_DIR =  os.path.join(AFFECTNET_DIR, "test_videos")
TEST_VIDEO_DIR =  "C:\\Users\\arman\\Desktop\\emotion_recognition\\data\\test_videos"

# Define the corresponding categorical emotions of AffectNet
# 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt
AFFECTNET_CAT_EMOT = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
FROM_EMOT_TO_ID = {emot: i for i, emot in enumerate(AFFECTNET_CAT_EMOT)}
NUMBER_OF_EMOT = len(AFFECTNET_CAT_EMOT)
EMOT_COLORS = ['darkgrey', 'yellow', 'dodgerblue', 'orange', 'fuchsia', 'deeppink', 'red', 'limegreen'] # https://matplotlib.org/stable/gallery/color/named_colors.html


# Define columns for the interim labels
INTERIM_COLUMNS_PAMI = ['path','orig_db', 'img_size', 'people', 'bbox', 'label_cat', 'label_cont', 'gender', 'age']
INTERIM_COLUMNS_AFFECTNET = ['path', 'cat_emot', 'valence', 'arousal']

# Define columns for the processed labels and processed properties
PIXELS_PER_IMAGE = 224 * 224

