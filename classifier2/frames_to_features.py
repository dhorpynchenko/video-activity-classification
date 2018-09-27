import argparse
import os
import tensorflow as tf
import numpy as np
from utils import bytes_feature

from classifier2.model import FrameFeaturesExtractor

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Extract features from frames')

parser.add_argument('--input_dir', required=True,
                    metavar="/path/to/json/",
                    help='Path to directory with activity/video_name/frame files')
parser.add_argument('--output_dir', required=True,
                    metavar="/path/to/json/",
                    help='Path to directory for output')

args = parser.parse_args()


def _convert_to_example(embeddings):
    return [tf.train.Example(features=tf.train.Features(feature={'image/features': bytes_feature(feature.tobytes())}))
            for feature in embeddings]


extractor = FrameFeaturesExtractor()

activity_dict = dict()

activity_list = os.listdir(args.input_dir)
for i, activity in enumerate(activity_list):
    activity_dict[activity] = i

    output_activity_dir = os.path.join(args.output_dir, activity)
    if not os.path.exists(args.output_dir):
        os.makedirs(output_activity_dir)

    curr_activ_path = os.path.join(args.input_dir, activity)
    video_list = os.listdir(curr_activ_path)
    for video_name in video_list:
        video_output_path = os.path.join(output_activity_dir, "{}.tfrecord".format(video_name))

        writer = tf.python_io.TFRecordWriter(video_output_path)

        current_video_path = os.path.join(curr_activ_path, video_name)

        reader = tf.python_io.tf_record_iterator(current_video_path)

        for data in reader:
            input_example = tf.train.Example()
            input_example.ParseFromString(data)

            height = int(input_example.features.feature['image/height']
                         .int64_list
                         .value[0])

            width = int(input_example.features.feature['image/width']
                        .int64_list
                        .value[0])

            image = np.frombuffer(input_example.features.feature['image/array'].bytes_list.value[0], np.uint8)
            image = image.reshape((width, height, 3))

            features = extractor.extract_features(image)
            examples = _convert_to_example(features)
            for item in examples:
                writer.write(item.SerializeToString())

        writer.close()
