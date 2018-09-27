import argparse
import os
import tensorflow as tf
from classifier2.model import RNNModel

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Extract features from frames')

parser.add_argument('--video', required=True,
                    metavar="/path/to/json/",
                    help='Path to video to classify')

parser.add_argument('--rnn_weights', required=True,
                    metavar="/path/to/json/",
                    help='Path to weights')

parser.add_argument('--activities_classes_names', required=True,
                    metavar="/path/to/json/",
                    help='Path to file with mappings id:class_name')

parser.add_argument('--mrcnn_object_classes', required=True,
                    metavar="/path/to/class_ids_file/",
                    help='Class ids files after training')

parser.add_argument('--mrcnn_weights',
                    metavar="/path/to/weights.h5",
                    help="Path to weights .h5 file")

args = parser.parse_args()

model = RNNModel(False)

