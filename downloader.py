import utils

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Load images')
parser.add_argument('--dataset_dir', required=True,
                    metavar="/path/to/coco/",
                    help='Directory of the MS-COCO dataset')
parser.add_argument('--dataset_config', required=True,
                    metavar="/path/to/json/",
                    help='Path to json file from labelbox')

args = parser.parse_args()

dataset = utils.ProjectDataset(args.dataset_config)
dataset.load_dataset(args.dataset_dir)
