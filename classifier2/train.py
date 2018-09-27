import argparse
import os
import tensorflow as tf
from classifier2.model import RNNModel
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Extract features from frames')

parser.add_argument('--input_dir', required=True,
                    metavar="/path/to/json/",
                    help='Path to directory with activity/video features files')

parser.add_argument('--output_dir', required=True,
                    metavar="/path/to/json/",
                    help='Path to directory for checkpoints save')

args = parser.parse_args()

model = RNNModel(True)

activity_list = os.listdir(args.input_dir)
for i, activity in enumerate(activity_list):

    output_activity_dir = os.path.join(args.output_dir, activity)
    if not os.path.exists(args.output_dir):
        os.makedirs(output_activity_dir)

    curr_activ_path = os.path.join(args.input_dir, activity)
    video_list = os.listdir(curr_activ_path)
    for video_name in video_list:
        current_video_path = os.path.join(curr_activ_path, video_name)
        reader = tf.python_io.tf_record_iterator(current_video_path)

        frames = list()

        for data in reader:
            input_example = tf.train.Example()
            input_example.ParseFromString(data)
            frames.append(np.frombuffer(input_example.features.feature['image/features'].bytes_list.value[0], np.float32))

