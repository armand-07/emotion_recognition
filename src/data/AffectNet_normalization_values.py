import argparse
import pandas as pd
import os
import torch
import torchvision.transforms.v2 as transforms
from tqdm import tqdm

from src import PROCESSED_AFFECTNET_DIR
from src.data.dataset import AffectNetDatasetValidation

PIXELS_IMAGE = 256 * 256

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasplits', type=str, default='all', help='The datasplits that will be usen to compute the metrics')
    return parser.parse_args() 

def main():
    # Compute metrics for the specified datasplits
    args = parse_args()
    if args.datasplits == 'all':
        datasplits = ['train', 'val', 'test']
    elif args.datasplits == 'train':
        datasplits = ['train']
    elif args.datasplits == 'val':
        datasplits = ['val']
    elif args.datasplits == 'test':
        datasplits = ['test']

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using device:", device)
        
    # Variables used to compute the metrics
    count = 0
    psum = torch.tensor([0.0, 0.0, 0.0]).to(device)
    psum_sq = torch.tensor([0.0, 0.0, 0.0]).to(device)
    img_transforms = transforms.ToDtype(torch.float64, scale = True) # Convert the image to float32 and scale it to [0,1]

    # Compute metrics
    for datasplit in datasplits:
        print("Working on datasplit:", datasplit)
        df = pd.read_pickle(os.path.join(PROCESSED_AFFECTNET_DIR, datasplit + '.pkl'))
        count += df.shape[0]
        print("Total entries:", df.shape[0])
        dataset= AffectNetDatasetValidation(annotations_path=os.path.join(PROCESSED_AFFECTNET_DIR, datasplit + '.pkl'), img_transforms = img_transforms)
        image_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True)
        for inputs in tqdm(image_loader):
            imgs = inputs[1].to(device)
            psum += imgs.sum(axis=[0, 2, 3])
            psum_sq += (imgs**2).sum(axis=[0, 2, 3])

    # Compute final metrics
    pixel_count = count * PIXELS_IMAGE
    total_mean = psum / pixel_count
    total_var = (psum_sq / pixel_count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    total_mean = total_mean.cpu()
    total_std = total_std.cpu()

    # Show output
    print("Computed mean: " + str(total_mean))
    print("Computed std:  " + str(total_std))

    # Save results 
    torch.save({'mean': total_mean, 'std': total_std}, os.path.join (PROCESSED_AFFECTNET_DIR, 'dataset_normalization_values.pt'))
    


if __name__ == '__main__':
    main()