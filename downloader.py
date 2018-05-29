import labelbox

import argparse
import os

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

if not os.path.exists(os.path.abspath(args.dataset_dir)):
    os.makedirs(os.path.abspath(args.dataset_dir))

dataset = labelbox.ProjectDataset(args.dataset_config)
dataset.load_dataset(args.dataset_dir)
