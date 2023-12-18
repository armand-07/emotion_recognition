from pathlib import Path


# Define global variables
ROOT_DIR = Path(Path(__file__).resolve().parent.parent)
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim"



# Define columns for the interim labels
INTERIM_COLUMNS = ['path','orig_db', 'img_size', 'people', 'bbox', 'label_cat', 'label_cont', 'gender', 'age']