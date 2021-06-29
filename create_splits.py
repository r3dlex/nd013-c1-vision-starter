import argparse
import glob
import os
import shutil

import numpy as np

from utils import get_module_logger

def get_train_validation_test_split(dataset_size: int, train_split_percentage: int = 75, validate_split_percentage: int = 15) -> (int, int, int):
    #calculates the splits keeping the arithmetic in int
    train_count = train_split_percentage * dataset_size // 100
    validate_count = validate_split_percentage * dataset_size // 100

    #test set is the remainder of the other two
    test_count = dataset_size - train_count - validate_count

    return train_count, validate_count, test_count

def split(data_dir: str):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    files = [filename for filename in glob.glob(f'{data_dir}/*.tfrecord')]
    np.random.shuffle(files)
    
    train_count, validate_count, t = get_train_validation_test_split(len(files))

    return np.split(files, [train_count, train_count + validate_count])


def move_files_into_dir(target_dir: str, files: [str]):
    """
    Move files into directory
    """
    os.makedirs(target_dir, exist_ok=True)
    for filepath in files:
        shutil.move(filepath, target_dir)

def move_train_validation_test_data(data_dir: str, train_files: [str], validation_files: [str], test_files: [str]):
    """
    Moves train, validation and test data into their folders
    """
    named_datasets = {'train': train_files, 'validation': validation_files, 'test': test_files}

    for dataset_name, dataset in named_datasets.items():
        target_dir = os.path.join(data_dir, dataset_name)
        move_files_into_dir(target_dir, dataset)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    train_files, validation_files, test_files = split(args.data_dir)
    move_train_validation_test_data(args.data_dir, train_files, validation_files, test_files)