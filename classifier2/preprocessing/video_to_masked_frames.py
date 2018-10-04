import argparse
import os
import shutil
import tempfile

import tensorflow as tf

from classifier2.model.model import ModelConfig
from classifier2.preprocessing.preprocessing import MRCNNPreprocessing
from utils import int64_feature, bytes_feature

MAX_VIDEOS_PER_CLASS = 100


def _convert_to_example(image, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(image.shape[1]),
        'image/width': int64_feature(image.shape[0]),
        'image/ids': int64_feature(label),
        # 'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        # 'image/channels': _int64_feature(channels),
        # 'image/class/label': _int64_feature(label),
        # 'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
        # 'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        # 'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/array': bytes_feature(image.tobytes(order='C'))}))
    return example


def main(args):
    config = ModelConfig.from_file(args.model_config)
    prepr = MRCNNPreprocessing(args.classes, args.model, config.sequence_length, config.frame_size)

    activity_list = os.listdir(args.input_dir)
    for i, activity in enumerate(activity_list):

        output_activity_dir = os.path.join(args.output_dir, activity)
        if not os.path.exists(output_activity_dir):
            os.makedirs(output_activity_dir)

        curr_activ_path = os.path.join(args.input_dir, activity)
        video_list = os.listdir(curr_activ_path)
        videos_processed = len(os.listdir(output_activity_dir))
        for video_name in video_list:

            if videos_processed >= MAX_VIDEOS_PER_CLASS:
                break

            curr_video_path = os.path.join(curr_activ_path, video_name)
            video_output_path = os.path.join(output_activity_dir, "{}.tfrecord".format(os.path.splitext(video_name)[0]))

            if not os.path.exists(curr_activ_path):
                os.makedirs(curr_activ_path)

            if os.path.exists(video_output_path):
                continue

            print("Proccessing video %s from activity %s" % (video_name, activity))

            fd, tfile_path = tempfile.mkstemp()
            try:

                writer = tf.python_io.TFRecordWriter(tfile_path)
                count = 0
                for ids, image in prepr.process_video(curr_video_path):
                    example = _convert_to_example(image, ids)
                    # print("Converting took %s ms" % (datetime.now() - time))
                    writer.write(example.SerializeToString())

                    count += 1
                print("Count %s" % count)
                writer.close()
                shutil.copyfile(tfile_path, video_output_path)
            finally:
                os.close(fd)
                os.remove(tfile_path)
            videos_processed += 1


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract features from frames')

    parser.add_argument('--input_dir', required=True,
                        metavar="/path/to/json/",
                        help='Path to directory with activity/activity/video files')
    parser.add_argument('--output_dir', required=True,
                        metavar="/path/to/json/",
                        help='Path to directory for output')
    parser.add_argument('--classes', required=True,
                        metavar="/path/to/class_ids_file/",
                        help='Class ids files after training')
    parser.add_argument('--model',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--model_config',
                        metavar="/path/to/weights.h5",
                        help="Path to model config file")

    args = parser.parse_args()
    main(args)
