import argparse
import os
import tensorflow as tf
import numpy as np

import utils
from utils import bytes_feature

from classifier2.model import FrameFeaturesExtractor

CLASS_IDS_FILENAME = "class_ids.txt"
INFORMATION_FILENAME = "info.txt"


def _convert_to_example(embeddings):
    return [tf.train.Example(features=tf.train.Features(feature={'image/features': bytes_feature(feature.tobytes())}))
            for feature in embeddings]


class Information:

    def __init__(self) -> None:
        self.size = 0
        self.embedding_sizes = []


def main(args):
    extractor = FrameFeaturesExtractor()

    activity_dict = dict()
    information = Information()

    activity_list = os.listdir(args.input_dir)
    for i, activity in enumerate(activity_list):
        activity_dict[activity] = i

        output_activity_dir = os.path.join(args.output_dir, activity)
        if not os.path.exists(output_activity_dir):
            os.makedirs(output_activity_dir)

        utils.clear_folder(output_activity_dir)

        curr_activ_path = os.path.join(args.input_dir, activity)
        video_list = os.listdir(curr_activ_path)
        for video_name in video_list:
            information.size += 1
            video_output_path = os.path.join(output_activity_dir,
                                             "{}_features.tfrecord".format(os.path.splitext(video_name)[0]))

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

                ids = input_example.features.feature['image/ids'].int64_list.value

                image = np.frombuffer(input_example.features.feature['image/array'].bytes_list.value[0], np.uint8)
                image = image.reshape((width, height, 3))

                features = extractor.extract_features(image, ids)
                examples = _convert_to_example(features)
                for item in examples:
                    writer.write(item.SerializeToString())

                information.embedding_sizes = len(features[0])

            writer.close()

    class_ids_file = os.path.join(args.output_dir, CLASS_IDS_FILENAME)
    with open(class_ids_file, "w") as f:
        for class_name in activity_dict.keys():
            f.write("{}\t{}\n".format(activity_dict[class_name], class_name))

    utils.save_obj(information, os.path.join(args.output_dir, INFORMATION_FILENAME))


if __name__ == '__main__':
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
    main(args)
