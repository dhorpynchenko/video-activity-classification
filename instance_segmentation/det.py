# Program for instance segmentation by Adil C.P
#To be used with Mask RCNN


################################################################################
"""Import Required Libraries"""
################################################################################

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
import cv2
import train

################################################################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable tensorflow logs

################################################################################
"""All required directories"""
################################################################################

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
my_model=('/home/adil_cp/Downloads/mask_rcnn_training_0026_final.h5')
IMAGE_DIR = ('/home/adil_cp/Desktop/V&P')
json_file='/home/adil_cp/Desktop/labelbox/beer_pong_labels.json'

################################################################################
"""Create dataset from exsisting dataset directory"""
################################################################################

dataset=utils1.ProjectDataset
dataset_info = utils1.ProjectDataset(json_file)

################################################################################
"""Inference class for the configuration details"""
################################################################################
class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    NAME = "training"
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8
    #NUM_CLASSES = 1 + 3
    def __init__(self, dataset: utils1.ProjectDataset):
        Config.NUM_CLASSES = len(dataset.classes) + 1
        super().__init__()

config = InferenceConfig(dataset=dataset_info)
#config.display()

################################################################################
"""Create model object in inference mode and Load Dataset"""
################################################################################

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(my_model, by_name=True)
dataset=train.TrainDataset(dataset=dataset_info,dataset_dir='/home/adil_cp/Desktop/tst')
dataset.load_data_train
dataset.prepare()

################################################################################
"""Setup the One hot encoding"""
################################################################################

def class_names(dataset: utils1.ProjectDataset):
    data=sorted(dataset.classes)
    background=['BG']
    class_names=background+data
    return data


class_names = class_names(dataset=dataset_info)
print(class_names)

################################################################################
"""Predictions"""
################################################################################

file_names = next(os.walk(IMAGE_DIR))[2]
for i in range(1):#Todo as per the video frames
    image = skimage.io.imread(os.path.join(IMAGE_DIR, 'tst.jpg'))
    results = model.detect([image], verbose=1)
    r = results[0]
    a=visualize.display_instances(i, image , r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
