import argparse
import itertools
import os
from random import shuffle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import plot_utils
import utils
from sklearn.model_selection import train_test_split

from classifier2.model.model import ModelConfig, ModelFactory
from classifier2.preprocessing.features import FrameFeaturesExtractor
from classifier2.preprocessing.preprocessing import NoPreprocessing
import datetime

TOTAL_EPOCHS = 60
ACTIVITY_ID_NAME_MAPPING_FILENAME = "activities_ids.txt"

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract features from frames')

    parser.add_argument('--input_dir', required=True,
                        metavar="/path/to/json/",
                        help='Path to directory with preprocessed videos as tfrecords')

    parser.add_argument('--model_dir', required=True,
                        metavar="/path/to/json/",
                        help='Path to directory for models. Inside model dir will be created')

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

    # id:name
    rev_activity_dict = utils.make_reversed_dict(activity_dict)

    x_train, x_valid = train_test_split(data_items_paths, test_size=0.25, shuffle=True)


    def generate_batch(dataset_path):
        shuffle(dataset_path)
        features_size = FrameFeaturesExtractor.OUTPUT_SIZE
        position = 0
        while True:
            if position >= len(dataset_path):
                break
            x_batch = np.zeros((config.batch_size, config.sequence_length, features_size))
            y_batch = np.zeros(config.batch_size)
            indexes = np.arange(position, position + config.batch_size)
            for i in range(len(indexes)):
                index = indexes[i]
                if index >= len(dataset_path):
                    index = index - len(dataset_path)
                item = dataset_path[index]
                reader = tf.python_io.tf_record_iterator(item)
                frames = np.zeros((config.sequence_length, features_size))
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
            position += position + config.batch_size


    model_dir = os.path.join(args.model_dir, "act_classifier_{:%Y%m%dT%H%M}".format(datetime.datetime.now()))

    model = ModelFactory.new_keras_model(config, FrameFeaturesExtractor.OUTPUT_SIZE, len(activity_dict))

    bar = tqdm(range(TOTAL_EPOCHS), "Epoch", TOTAL_EPOCHS)

    display_data = {}
    history = {"loss": [], "acc": []}

    weights_dir = os.path.join(model_dir, "training")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    for i in bar:
        loss = 0
        total = 0
        for x_batch, y_batch in generate_batch(x_train):
            loss += model.train(x_batch, y_batch)
            total += 1

        loss = loss / total
        display_data["loss"] = loss
        history["loss"].append(loss)

        total = 0
        correct = 0
        for x_v_batch, y_v_batch in itertools.islice(generate_batch(x_valid), None, len(x_valid) // config.batch_size):
            prediction = model.classify(x_v_batch)
            for j in range(len(prediction)):
                total += 1
                if prediction[j] == y_v_batch[j]:
                    correct += 1

        acc = correct / total
        display_data["eval_acc"] = acc
        history["acc"].append(acc)
        bar.set_postfix(display_data)

        model.save_weights(weights_dir, i)

    model.save(model_dir)
    utils.save_class_ids(activity_dict, os.path.join(args.output_dir, ACTIVITY_ID_NAME_MAPPING_FILENAME))

    index = np.argmax(history['acc'])
    comment = "Max accuracy {} after {} epoch".format(history['acc'][index], index)
    print(comment)

    plot_utils.create_plot(history['acc'], history['loss'], os.path.join(model_dir, "results.png"), True, comment)
