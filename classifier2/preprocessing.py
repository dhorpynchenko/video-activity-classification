import argparse
import os
import shutil
import tempfile

import cv2
from PIL import Image

import utils
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import tensorflow as tf
import numpy as np
from utils import int64_feature, bytes_feature
from datetime import datetime
from classifier2.model import ModelConfig

MAX_VIDEOS_PER_CLASS = 100


class PreprocConfig(Config):
    NAME = "preproc"
    IMAGES_PER_GPU = 1  # 1 reduces training time but gives an error https://github.com/matterport/Mask_RCNN/issues/521
    DETECTION_MIN_CONFIDENCE = 0.6

    def __init__(self, classes_ids):
        Config.NUM_CLASSES = len(classes_ids)
        super().__init__()


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


class Preprocessing:

    def __init__(self, mrcnn_classes_file, mrcnn_weights, ) -> None:
        # Mask-RCNN model
        classes_ids = utils.load_class_ids(mrcnn_classes_file)
        config = PreprocConfig(classes_ids)

        # Create model object in inference mode.
        self.model = MaskRCNN(mode="inference", model_dir="./log", config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(mrcnn_weights, by_name=True, exclude=[])

    def process_image_mrcnn(self, image):
        results = self.model.detect([image], verbose=0)
        # print("Detecting took %s ms" % (datetime.now() - time))
        # time = datetime.now()
        r = results[0]
        ids = r['class_ids']
        maschere = r["masks"]
        return ids, maschere

    def apply_masks_to_image(self, image, masks):
        for r in range(min(masks.shape[0], image.shape[0])):
            for c in range(min(masks.shape[1], image.shape[1])):
                if not np.any(masks[r, c]):
                    image[r][c] = (0, 0, 0)

    def process_video(self, path):
        vidcap = None
        try:
            vidcap = cv2.VideoCapture(path)
            if not vidcap.isOpened():
                print("could not open %s" % path)
                return

            length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vidcap.get(cv2.CAP_PROP_FPS)

            print("Total frames %s, %s x %s, fps %s" % (length, width, height, fps))

            interval = max(1, length // ModelConfig.SEQUENCE_LENGTH)

            count = 0
            skipping = 0
            while count < ModelConfig.SEQUENCE_LENGTH:
                success, image = vidcap.read()
                if not success:
                    break

                # image = cv2.resize(image, (OUTPUT_SIZE, int(image.shape[1] * OUTPUT_SIZE / image.shape[0])))
                # time = datetime.now()
                image = cv2.resize(image, (ModelConfig.FRAME_SIZE, ModelConfig.FRAME_SIZE))
                ids, maschere = self.process_image_mrcnn(image)

                if skipping > 40:
                    break

                if len(ids) == 0:
                    skipping += 1
                    print("Skipping frame %s" % skipping)
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, (count * interval) + (skipping * int(fps / 2)))
                    continue

                skipping = 0
                print("Taking frame %s" % count)
                # Apply mask to original image
                self.apply_masks_to_image(image, maschere)
                yield ids, image
                count += 1
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, (count * interval))

        finally:
            if vidcap is not None:
                vidcap.release()


def main(args):
    prepr = Preprocessing(args.classes, args.model)

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

    args = parser.parse_args()
    main(args)
