import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils as mrcnn_utils
import coco
import utils
import utils1
import model as modellib
import visualize
import train
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Root directory of the project
ROOT_DIR = os.getcwd()
### ball ['BG', 'snooker_table', 'cue_sticks', 'billiard_balls', 'hand', 'rubix_cube']
# ['BG', 'snooker_table', 'billiard_balls', 'hand' 'rubix_cube', 'cue_sticks']


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs123")
asd=('/home/adil_cp/Music/training20180518T0219/mask_rcnn_training_0010.h5')
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_training_0002.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
dataset=utils1.ProjectDataset
"""class InferenceConfig(Config):
    NAME = "inference"
    IMAGES_PER_GPU = 1  # Reduces training time

    def __init__(self,dataset):
        print("######################"+'\n')
        print(dataset)
        Config.NUM_CLASSES = len(dataset.classes) + 1
        super().__init__()
        self.pr_dataset = dataset
        print (Config.NUM_CLASSES)"""
# Directory of images to run detection on
IMAGE_DIR1 = ('/home/adil_cp/Desktop/V&P')
IMAGE_DIR = ("/home/adil_cp/Documents/projects/vision/VPproject/frame")
json_file='/home/adil_cp/Desktop/labelbox/labelbox1.json'
dataset_info = utils1.ProjectDataset(json_file)
class InferenceConfig(train.TrainConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8
config = InferenceConfig()
#config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(asd, by_name=True)
class_names1=list()
dataset=train.TrainDataset(dataset=dataset_info,dataset_dir='/home/adil_cp/Desktop/tst')
dataset.load_data_train
dataset.prepare()
for i, info in enumerate(dataset.class_info):
    #print(info['name'])
    class_names1.append(info['name'])
#print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
#class_names = ['BG','rubix_cube', 'snooker_table','billiard_balls','cue_sticks','hand']
#dataset.
print(class_names1)

file_names = next(os.walk(IMAGE_DIR1))[2]
print(len(file_names))

for i in range(1):

    image = skimage.io.imread(os.path.join(IMAGE_DIR1, 'fa.jpg'))

# Run detection
    results = model.detect([image], verbose=1)
    #print (results[0])

# Visualize results
    r = results[0]
    a=visualize.display_instances(i, image , r['rois'], r['masks'], r['class_ids'],
                            class_names1, r['scores'])
    #cv2.imwrite("frame.jpg", a)
