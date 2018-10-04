import argparse
import itertools
import os
from random import shuffle

import numpy as np
import tensorflow as tf

import utils
from sklearn.model_selection import train_test_split

from classifier2.model.model import ModelConfig, RNNTensorflowModel
from classifier2.preprocessing.features import FrameFeaturesExtractor
from classifier2.preprocessing.preprocessing import NoPreprocessing

TOTAL_EPOCHS = 25

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Extract features from frames')

parser.add_argument('--input_dir', required=True,
                    metavar="/path/to/json/",
                    help='Path to directory with videos')

parser.add_argument('--model_dir', required=True,
                    metavar="/path/to/json/",
                    help='Path to directory for checkpoints save')

parser.add_argument('--init_config', required=True,
                    metavar="/path/to/json/",
                    help='Path to initial config file')

args = parser.parse_args()

config = ModelConfig.from_file(args.init_config)
preproc = NoPreprocessing(config.frame_size)
extractor = FrameFeaturesExtractor()

activities = os.listdir(args.input_dir)

data_items_paths = list()
items_activity = dict()
# name:id
activity_dict = {}
for activity_name in activities:

    curr_act_dir = os.path.join(args.input_dir, activity_name)

    if not os.path.isdir(curr_act_dir):
        continue

    activity_dict[activity_name] = len(activity_dict)
    items = os.listdir(curr_act_dir)

    for item in items:
        item_path = os.path.join(curr_act_dir, item)
        data_items_paths.append(item_path)
        items_activity[item_path] = activity_name

shuffle(data_items_paths)

# id:name
rev_activity_dict = utils.make_reversed_dict(activity_dict)

x_train, x_valid = train_test_split(data_items_paths, test_size=0.25)


def generate_batch(dataset_path):
    features_size = FrameFeaturesExtractor.OUTPUT_SIZE
    while True:
        x_batch = np.zeros((config.batch_size, config.sequence_length, features_size))
        y_batch = np.zeros(config.batch_size)
        index = np.random.choice(len(dataset_path), config.batch_size)
        for i in range(len(index)):
            item = dataset_path[index[i]]
            reader = tf.python_io.tf_record_iterator(item)
            frames = np.zeros((config.sequence_length, features_size))
            # print("Reading %s" % item)
            count = 0
            for data in reader:
                if count >= config.sequence_length:
                    break
                input_example = tf.train.Example()
                input_example.ParseFromString(data)

                features = np.frombuffer(input_example.features.feature['image/features'].bytes_list.value[0],
                                         np.float32)
                frames[count] = features
                count += 1
            reader.close()
            x_batch[i] = frames
            y_batch[i] = activity_dict[items_activity[item]]
        yield np.asarray(x_batch), np.asarray(y_batch)


# model = RNNKerasModel(FrameFeaturesExtractor.OUTPUT_SIZE, len(activity_dict), ModelConfig.SEQUENCE_LENGTH, is_training=True)
model = RNNTensorflowModel(FrameFeaturesExtractor.OUTPUT_SIZE, FrameFeaturesExtractor.OUTPUT_SIZE, len(activity_dict), is_training=True)
for i in range(TOTAL_EPOCHS):
    print("Epoch %s of %s" % (i, TOTAL_EPOCHS))
    for x_batch, y_batch in itertools.islice(generate_batch(x_train), None, len(x_train) // config.batch_size):
        model.train(x_batch, y_batch)

    total = 0
    correct = 0
    for x_v_batch, y_v_batch in itertools.islice(generate_batch(x_valid), None, len(x_valid) // config.batch_size):
        prediction = model.classify(x_v_batch)
        for j in range(len(prediction)):
            total += 1
            if prediction[j] == y_v_batch[j]:
                correct += 1
    print("Evaluation accuracy %s" % (correct / total))

model.save(args.model_dir)