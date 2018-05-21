# Program for instance segmentation by Adil C.P
#To be used with Mask RCNN

#python det.py --model=/home/adil_cp/Downloads/mask_rcnn_training_0026_final.h5
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
import model as modellib
import visualize
import cv2
import argparse
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
from os import listdir
from os.path import isfile, join
import segmentation_support

################################################################################
"""Required Arguments"""
################################################################################

parser = argparse.ArgumentParser(
    description='Instance Segmentation using Mask-RCNN')
parser.add_argument('--model', required=True,
                    default="/home/adil_cp/Downloads/mask_rcnn_training_0026_final.h5",
                    metavar="/path/to/weights.h5",
                    help="Load the trained model")
parser.add_argument('--img_dir', required=False,
                    default="/home/adil_cp/Desktop/V&P",
                    metavar="/path/to/logs/",
                    help='path to the image directory')
parser.add_argument('--json_file', required=False,
                    default="/home/adil_cp/Desktop/labelbox/beer_pong_labels.json",
                    metavar="/path/to/json/",
                    help='Path to json file from labelbox')
parser.add_argument('--video_file', required=False,
                    default="/home/adil_cp/Desktop/beer.mp4",
                    metavar="/path/to/json/",
                    help='Path to json file from labelbox')
parser.add_argument('--frame_folder', required=False,
                    default="/home/adil_cp/Desktop/beer/",
                    metavar="/path/to/json/",
                    help='Path to json file from labelbox')
parser.add_argument('--Processed_folder', required=False,
                    default="/home/adil_cp/Desktop/beer/Processed/",
                    metavar="/path/to/json/",
                    help='Path to json file from labelbox')
parser.add_argument('--masks', required=False,
                    default="/home/adil_cp/Desktop/beer/all_masks/",
                    metavar="/path/to/json/",
                    help='Path to json file from labelbox')
parser.add_argument('--b_masks', required=False,
                    default="/home/adil_cp/Desktop/beer/b_masks/",
                    metavar="/path/to/json/",
                    help='Path to json file from labelbox')
args = parser.parse_args()

################################################################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable tensorflow logs

################################################################################
"""All required directories"""
################################################################################

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
my_model=args.model
IMAGE_DIR = args.img_dir
json_file=args.json_file

################################################################################
"""Video to Frames"""
################################################################################

vidcap = cv2.VideoCapture(args.video_file)
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite(args.frame_folder+"frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

################################################################################
"""Create dataset from exsisting dataset directory"""
################################################################################

dataset=segmentation_support.ProjectDataset
dataset_info = segmentation_support.ProjectDataset(json_file)

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
    def __init__(self, dataset: segmentation_support.ProjectDataset):
        Config.NUM_CLASSES = len(dataset.classes) + 1
        super().__init__()

config = InferenceConfig(dataset=dataset_info)
#config.display()

################################################################################
"""Load model"""
################################################################################

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(my_model, by_name=True)

################################################################################
"""Setup the One hot encoding"""
################################################################################

def class_names(dataset: segmentation_support.ProjectDataset):
    data=sorted(dataset.classes)
    background=['BG']
    class_names=background+data
    return data


class_names = class_names(dataset=dataset_info)
print(class_names)

################################################################################
"""Predictions"""
################################################################################

file_names = next(os.walk(args.frame_folder))[2]
for i in range(0,len(file_names),1):#Todo as per the video frames
    image = skimage.io.imread(os.path.join(args.frame_folder, 'frame'+str(i)+'.jpg'))
    results = model.detect([image], verbose=1)
    r = results[0]
    a=visualize.display_instances(i, image , r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

################################################################################
"""Binary Image"""
################################################################################ù

cv2.CV_LOAD_IMAGE_COLOR = 0
#img.save('greyscale1.png')
dir=args.b_masks
mypath=args.masks
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for i in range(0,len(onlyfiles),1):
    im_gray = cv2.imread(mypath+onlyfiles[i], cv2.CV_LOAD_IMAGE_COLOR)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 10
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(dir+onlyfiles[i], im_bw)

################################################################################
"""create final VIDEO"""
################################################################################

IMAGE_DIR=(args.Processed_folder)
def make_video(images, outimg=None, fps=5, size=None,
               is_color=True, format="XVID"):

    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(IMAGE_DIR+'final.mp4', fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid
file_names = next(os.walk(IMAGE_DIR))[2]
print((file_names))
file_names1=list()
for i in range(len(file_names)):
    file_names1.append(IMAGE_DIR+'frame-%d.jpg' % i)
print(file_names1)
make_video(file_names1, outimg=None, fps=10, size=None,
               is_color=True, format="XVID")

################################################################################
"""create final Binary VIDEO"""
################################################################################

IMAGE_DIR=(args.b_masks)
onlyfiles = sorted([f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))])
def make_video(images, outimg=None, fps=5, size=None,
               is_color=True, format="XVID"):

    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(IMAGE_DIR+'final_binary.mp4', fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid
file_names = next(os.walk(IMAGE_DIR))[2]
print((file_names))
file_names1=list()
for i in range(len(file_names)):
    file_names1.append(IMAGE_DIR+onlyfiles[i])
print(file_names1)
make_video(file_names1, outimg=None, fps=10, size=None,
               is_color=True, format="XVID")

################################################################################
"""create final mask VIDEO"""
################################################################################

IMAGE_DIR=(args.masks)
onlyfiles = sorted([f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))])
def make_video(images, outimg=None, fps=5, size=None,
               is_color=True, format="XVID"):

    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(IMAGE_DIR+'final_binary.mp4', fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid
file_names = next(os.walk(IMAGE_DIR))[2]
print((file_names))
file_names1=list()
for i in range(len(file_names)):
    file_names1.append(IMAGE_DIR+onlyfiles[i])
print(file_names1)
make_video(file_names1, outimg=None, fps=10, size=None,
               is_color=True, format="XVID")
################################################################################
