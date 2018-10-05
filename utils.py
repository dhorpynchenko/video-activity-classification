import os
import pickle
import shutil

import tensorflow as tf
import numpy as np


def load_required_classes(file):
    class_names = []
    with open(file) as file:
        for line in file.readlines():
            if line and not line.startswith('#'):
                class_names.append(line.strip("\n"))

    return class_names


def load_class_ids(file):
    classes = dict()
    with open(file) as file:
        for line in file.readlines():
            parts = line.split("\t")
            classes[int(parts[0])] = parts[1].strip("\n")
    return classes


def save_class_ids(class_ids: dict, file):
    """

    :param class_ids: name:id
    :param file: file to save
    :return:
    """
    with open(file, "w") as f:
        for class_name in class_ids.keys():
            f.write("{}\t{}\n".format(class_ids[class_name], class_name))


def make_reversed_dict(dictionary: dict):
    return dict(zip(dictionary.values(), dictionary.keys()))


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, 0)


def load_obj(name):
    if os.path.exists(name):
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        return None


def clear_folder(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
