import argparse
import os

import utils
from classifier2.model.model import ModelFactory
from classifier2.train import ACTIVITY_ID_NAME_MAPPING_FILENAME

# Parse command line arguments
from classifier2.preprocessing.features import FrameFeaturesExtractor
from classifier2.preprocessing.preprocessing import MRCNNPreprocessing

parser = argparse.ArgumentParser(
    description='Classify video/videos provided')

parser.add_argument('--video', required=True,
                    metavar="/path/to/json/",
                    help='Path to video to classify')

parser.add_argument('--model_dir', required=True,
                    metavar="/path/to/json/",
                    help='Path to model configs')

parser.add_argument('--mrcnn_object_classes', required=True,
                    metavar="/path/to/class_ids_file/",
                    help='Class ids files after training')

parser.add_argument('--mrcnn_weights',
                    metavar="/path/to/weights.h5",
                    help="Path to weights .h5 file")

args = parser.parse_args()

activity_classes = utils.load_class_ids(os.path.join(args.model_dir, ACTIVITY_ID_NAME_MAPPING_FILENAME))

model = ModelFactory.restore_tf_model(args.model_dir)
preprocessing = MRCNNPreprocessing(args.mrcnn_object_classes, args.mrcnn_weights, model.model_config.sequence_length,
                                   model.model_config.frame_size)
extractor = FrameFeaturesExtractor()

frames = []
ids = []

for frame_ids, frame in preprocessing.process_video(args.video):
    frames.append(extractor.extract_features(frame[0], frame_ids))
    ids.append(frame_ids)

activity_id = model.classify(frames)
activity_name = activity_classes[activity_id]

print("Video %s has %s activity" % (args.video, activity_name))
