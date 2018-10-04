import argparse
import os

import tensorflow as tf

import utils
from classifier2.model.model import ModelConfig
from classifier2.preprocessing.features import FrameFeaturesExtractor
from classifier2.preprocessing.preprocessing import NoPreprocessing
from utils import bytes_feature

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
    model_config = ModelConfig.from_file(args.model_config)
    extractor = FrameFeaturesExtractor()
    preproc = NoPreprocessing(model_config.frame_size)

    activity_dict = dict()
    information = Information()

    activity_list = os.listdir(args.input_dir)
    for i, activity in enumerate(activity_list):
        activity_dict[activity] = i

        output_activity_dir = os.path.join(args.output_dir, activity)
        if not os.path.exists(output_activity_dir):
            os.makedirs(output_activity_dir)

        curr_activ_path = os.path.join(args.input_dir, activity)
        video_list = os.listdir(curr_activ_path)
        for video_name in video_list:
            information.size += 1
            video_output_path = os.path.join(output_activity_dir,
                                             "{}_features.tfrecord".format(os.path.splitext(video_name)[0]))

            if os.path.exists(video_output_path):
                continue

            writer = tf.python_io.TFRecordWriter(video_output_path)

            current_video_path = os.path.join(curr_activ_path, video_name)
            print("Extracting features from video %s of activity %s" % (video_name, activity))

            for frame in preproc.process_video(current_video_path):

                features = extractor.extract_features(frame, None)
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

    parser.add_argument('--model_config', required=True,
                        metavar="/path/to/json/",
                        help='Path to model config file')

    args = parser.parse_args()
    main(args)
