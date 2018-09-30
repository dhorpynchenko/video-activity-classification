import argparse

import utils
from classifier2.model import RNNModel, FrameFeaturesExtractor
from classifier2.preprocessing import Preprocessing

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Classify video/videos provided')

parser.add_argument('--video', required=True,
                    metavar="/path/to/json/",
                    help='Path to video to classify')

parser.add_argument('--rnn_weights', required=True,
                    metavar="/path/to/json/",
                    help='Path to weights')

parser.add_argument('--activities_classes_names', required=True,
                    metavar="/path/to/json/",
                    help='Path to file with mappings id:class_name')

parser.add_argument('--mrcnn_object_classes', required=True,
                    metavar="/path/to/class_ids_file/",
                    help='Class ids files after training')

parser.add_argument('--mrcnn_weights',
                    metavar="/path/to/weights.h5",
                    help="Path to weights .h5 file")

args = parser.parse_args()

activity_classes = utils.load_class_ids(args.activities_classes_names)

preprocessing = Preprocessing(args.mrcnn_object_classes, args.mrcnn_weights)
extractor = FrameFeaturesExtractor()
model = RNNModel(FrameFeaturesExtractor.OUTPUT_SIZE, len(activity_classes), False)
model.restore(args.rnn_weights)

frames = []
ids = []

for frame_ids, frame in preprocessing.process_video(args.video):
    frames.append(extractor.extract_features(frame[0], frame_ids))
    ids.append(frame_ids)

activity_id = model.classify(frames)
activity_name = activity_classes[activity_id]

print("Video %s has %s activity" % (args.video, activity_name))
