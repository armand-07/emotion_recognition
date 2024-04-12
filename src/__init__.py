from pathlib import Path
import os


# Define path variables
ROOT_DIR = Path(Path(__file__).resolve().parent.parent)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
INTERIM_DATA_DIR = os.path.join(ROOT_DIR, "data", "interim")
INFERENCE_DIR = os.path.join(ROOT_DIR, "inference")

AFFECTNET_DIR = os.path.normpath("/mnt/gpid08/datasets/affectnet/") # substitute with the actual data path
RAW_AFFECTNET_DIR =  os.path.join(AFFECTNET_DIR, "raw") 
INTERIM_AFFECTNET_DIR = os.path.join(AFFECTNET_DIR, "interim")
PROCESSED_AFFECTNET_DIR = os.path.join(AFFECTNET_DIR, "processed")

# Define the corresponding categorical emotions of AffectNet
# 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt
AFFECTNET_CAT_EMOT = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"] 
NUMBER_OF_EMOT = len(AFFECTNET_CAT_EMOT)


# Define columns for the interim labels
INTERIM_COLUMNS_PAMI = ['path','orig_db', 'img_size', 'people', 'bbox', 'label_cat', 'label_cont', 'gender', 'age']
INTERIM_COLUMNS_AFFECTNET = ['path', 'cat_emot', 'valence', 'arousal']

# Define columns for the processed labels and processed properties
PIXELS_PER_IMAGE = 224 * 224

