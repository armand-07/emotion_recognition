import argparse
import pandas as pd
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src import PROCESSED_AFFECTNET_DIR, PIXELS_PER_IMAGE
from src.data.dataset import AffectNetDataset


def main(datasplits:list[str], output_path:str = PROCESSED_AFFECTNET_DIR):
    """
    Computes the normalization values for the AffectNet dataset. The values that are computed are the mean and the standard deviation
    for each channel. The values are saved in a file called dataset_normalization_values.pt. The values are computed using the formula:
        mean = sum(x) / N
        std = sqrt(sum(x^2) / N - mean^2)
        where x is the pixel value, N is the total number of pixels and the sum is over all the images in the dataset.
    The images are normalized in the range [0, 1] before computing the metrics.

    Params:
    datasplits (list[str]): The datasplits to be used to compute the normalization values
    Returns:
    None
    """
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using device:", device)
        
    # Initialize variables to compute the metrics
    count = 0
    psum = torch.tensor([0.0, 0.0, 0.0]).to(device)
    psum_sq = torch.tensor([0.0, 0.0, 0.0]).to(device)
    img_transforms = A.Compose([
    A.Normalize(mean=(0,0,0), std=(1,1,1)), # The values are converted from 0-255 to 0-1
    ToTensorV2()
    ])
    datasplit_sizes_df = pd.read_csv(os.path.join(output_path,"datasplit_sizes.csv"))

    # Compute metrics
    for datasplit in datasplits:
        print(datasplit_sizes_df[datasplit_sizes_df['Datasplit'] == datasplit]['Size'].values[0])
        count += datasplit_sizes_df[datasplit_sizes_df['Datasplit'] == datasplit]['Size'].values[0]
        dataset= AffectNetDataset(path = output_path, datasplit = datasplit, img_transforms = img_transforms)
        image_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
        for inputs in tqdm(image_loader):
            imgs = inputs[0].to(device)
            psum += imgs.sum(axis=[0, 2, 3])
            psum_sq += (imgs**2).sum(axis=[0, 2, 3])

    # Compute results
    pixel_count = count * PIXELS_PER_IMAGE
    total_mean = psum / pixel_count
    total_var = (psum_sq / pixel_count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    total_mean = total_mean.cpu()
    total_std = total_std.cpu()

    # Show output
    print("Computed mean: " + str(total_mean))
    print("Computed std:  " + str(total_std))

    # Save results 
    torch.save({'mean': total_mean, 'std': total_std}, os.path.join (output_path, 'dataset_normalization_values.pt'))


def parse_args():
    """Parses the arguments for the script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasplits', type=str, default='default', help='The datasplits that will be used to compute the metrics')
    return parser.parse_args() 


if __name__ == '__main__':
    # Compute metrics for the specified datasplits
    args = parse_args()
    if args.datasplits == 'default':
        datasplits = ['train', 'val']
    elif args.datasplits == 'all':
        datasplits = ['train', 'val', 'test']
    elif args.datasplits == 'train':
        datasplits = ['train']
    elif args.datasplits == 'val':
        datasplits = ['val']
    elif args.datasplits == 'test':
        datasplits = ['test']
        
    main(datasplits)