import argparse
import os
import tensorflow as tf

import utils
from classifier2.model import RNNModel, ModelConfig
import numpy as np
from classifier2.frames_to_features import CLASS_IDS_FILENAME, INFORMATION_FILENAME, Information
from random import shuffle
from tqdm import tqdm

TOTAL_EPOCHS = 100

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Extract features from frames')

parser.add_argument('--input_dir', required=True,
                    metavar="/path/to/json/",
                    help='Path to directory with activity/video features files')

parser.add_argument('--models_dir', required=True,
                    metavar="/path/to/json/",
                    help='Path to directory for checkpoints save')

args = parser.parse_args()

# id:name
activity_dict = utils.load_class_ids(os.path.join(args.input_dir, CLASS_IDS_FILENAME))
# name:id
rev_activity_dict = utils.make_reversed_dict(activity_dict)
dataset_info = utils.load_obj(os.path.join(args.input_dir, INFORMATION_FILENAME))


def read_dataset(batch_size, sequence_size):
    # Read all data items
    activity_list = os.listdir(args.input_dir)
    data_items = list()
    items_activity = dict()
    for activity_name in activity_list:

        curr_act_dir = os.path.join(args.input_dir, activity_name)

        if not os.path.isdir(curr_act_dir):
            continue

        current_act_id = rev_activity_dict[activity_name]

        items = os.listdir(curr_act_dir)

        for item in items:
            item_path = os.path.join(curr_act_dir, item)
            data_items.append(item_path)
            items_activity[item_path] = current_act_id

    shuffle(data_items)

    i = 0
    x_batch = []
    y_batch = []
    while i < len(data_items):
        x_batch.clear()
        y_batch.clear()
        while len(x_batch) < batch_size and i < len(data_items):
            video_path = data_items[i]
            vid_activ = items_activity[video_path]
            reader = tf.python_io.tf_record_iterator(video_path)

            frames = list()

            for data in reader:
                input_example = tf.train.Example()
                input_example.ParseFromString(data)
                frames.append(
                    np.frombuffer(input_example.features.feature['image/features'].bytes_list.value[0], np.float32))

            padding = np.zeros([dataset_info.embedding_sizes])
            while len(frames) < sequence_size:
                frames.append(padding)

            x_batch.append(frames)
            y_batch.append(vid_activ)
            i += 1
        yield np.asarray(x_batch), np.asarray(y_batch)


model = RNNModel(dataset_info.embedding_sizes, len(activity_dict), is_training=True)
bar = tqdm(range(TOTAL_EPOCHS))
for i in bar:
    for x_batch, y_batch in read_dataset(ModelConfig.BATCH_SIZE, ModelConfig.SEQUENCE_LENGTH):
        r = model.train(x_batch, y_batch)
