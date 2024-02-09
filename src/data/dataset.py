from torch.utils.data import Dataset
from src import PROCESSED_COLUMNS

class AffectNetDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.annotations = pd.read_pickle(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations.iloc[idx]
        img = cv2.imread(data[PROCESSED_COLUMNS[1]]) # Keep it in BGR format
        img = img / 255.0 # Convert the image from 0-255 range to 0-1 range
        img = img.astype(np.float32) # Convert the image to float32

        if self.transform:
            image = self.transform(image)

        cat_label = data[PROCESSED_COLUMNS[2]]
        cont_label = data[PROCESSED_COLUMNS[3]]

        return image, cat_label, cont_label # Return the image and the continuous and categorical labels