# -*- coding: utf-8 -*-
import pandas as pd
import scipy.io
import os

import sys
from os.path import dirname
sys.path.append(dirname(__file__))

from src import RAW_DATA_DIR


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    

if __name__ == '__main__':

    # Load the dataset
    mat_path = os.path.join(RAW_DATA_DIR, 'PAMI', 'annotations', 'Annotations.mat')
    mat = scipy.io.loadmat(mat_path)
    print(mat.keys())
    main()
