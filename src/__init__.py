from pathlib import Path
import os


# Define global variables
ROOT_DIR = Path(Path(__file__).resolve().parent.parent)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
INTERIM_DATA_DIR = os.path.join(ROOT_DIR, "data", "interim")
RAW_AffectNet_DIR = os.path.normpath("/mnt/gpid08/datasets/affectnet/") # substitute with the actual data path



# Define columns for the interim labels
INTERIM_COLUMNS = ['path','orig_db', 'img_size', 'people', 'bbox', 'label_cat', 'label_cont', 'gender', 'age']

# Define columns for the processed labels
PROCESSED_COLUMNS = ['face_photo_tensor', 'label_cat', 'label_cont']

# Define the corresponding categorical emotions of AffectNet
AFFECTNET_CAT_EMOT = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"] # 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt