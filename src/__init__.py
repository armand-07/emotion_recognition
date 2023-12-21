from pathlib import Path


# Define global variables
ROOT_DIR = Path(Path(__file__).resolve().parent.parent)
MODELS_DIR = ROOT_DIR / "models"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim"



# Define columns for the interim labels
INTERIM_COLUMNS = ['path','orig_db', 'img_size', 'people', 'bbox', 'label_cat', 'label_cont', 'gender', 'age']

# Define columns for the processed labels
PROCESSED_COLUMNS = ['face_photo_tensor', 'label_cat', 'label_cont']