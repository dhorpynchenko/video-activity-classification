import os
import sys
sys.path.append('/home/edoardo/Download/coco-master/PythonAPI')
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import cv2
from PIL import Image

# Root directory of the project
ROOT_DIR = "/home/students/Desktop/WD"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "zou.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join("/home/students/Desktop/SecondHand/secondHandDataset")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 8
    DETECTION_MIN_CONFIDENCE = 0.6

config = InferenceConfig()
#config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_bbox", "mrcnn_mask"])

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'Car', 'Longboard', 'Snowboarding Shoes', 'Skateboard', 'Bar', 'SkiPole', 'Mask', 'Tree', 'Street', 'Wall', 'Ski_helmet', 'Ramp', 'Rollerblade', 'Snow tube', 'Ski boots', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench','backpack', 'umbrella', 'handbag','frisbee', 'skis', 'snowboard', 'skateboard', 'sports ball', 'kite' ]
    

#['BG',  'spanner', 'screwdriver', 'hex_key', 'mesh', 'brush', 'guard', 'belt']

'''
                ['BG','person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


'''
def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def find_person(fig):
    indexList = [0,1,2,3,4,5,6,7,8]
    results = model.detect([fig], verbose=0)
    r = results[0]
    ax = get_ax(1)
    r = results[0]
    # ['BG', 'screwdriver', 'belt', 'guard', 'mesh', 'spanner', 'boh1', 'boh2'], r['scores']
    visualize.display_instances(fig, r['rois'], r['masks'], r['class_ids'], class_names , r['scores'], ax=ax, title="Predictions")

    ids = r['class_ids']
    ret = []
    for i in indexList:
        ret.append(np.count_nonzero(ids == i))
    return ids


def find_centroid(im):
    width, height = im.size
    XX, YY, count = 0, 0, 0
    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if im.getpixel((x, y)) == (255,255,255):
                XX += x
                YY += y
                count += 1
    return XX/count, YY/count

def compute_area(im):
    width, height = im.size
    area = 0
    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if im.getpixel((x, y)) == (255,255,255):
                area += 1
    return area



def centreAnalisi(fig, h, w):

    results = model.detect([fig], verbose=0)
    r = results[0]
    ids = r['class_ids']
    maschere = r["masks"]
    numMask = 0
    try:
        numMasks = maschere[0][0]
    except Exception as e:
        print(e)
        return 0

    maskRet = []
    for i in range(numMasks):
        img = np.zeros([h, w], dtype=np.bool)
        maskRet.append(img)
    for r in range(len(maschere)):
        for c in range(maschere[0]):
            for h in range(maschere[0][0]):
                if maschere[r][c][h]:
                    maskRet[h][r][c] = (255,255,255)

    '''
    for c in range(3):
    img[:, :, c] = np.where(indice == 1, 255, img[:, :, c])
    '''
    centroidi_ret= []
    aree = []
    for maskSingle in range(len(maskRet)):
        image = Image.fromarray(img, 'RGB')
        ww, hh = image.size
        aree.append(compute_area(image))
        ret = find_centroid(image, ww, hh)
        centroidi_ret.append(ret)
    return centroidi_ret, ids, aree



def test():
    file_names = next(os.walk(IMAGE_DIR))[2]
    for f in file_names:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, f))
        number = find_person(image)
        print(number)
