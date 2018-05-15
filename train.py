import skimage.io

from PIL import Image

from mrcnn.config import Config
from mrcnn import model as modellib, utils as mrcnn_utils
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)
import utils
import numpy as np

EVAL_PART = 0.1


class TrainConfig(Config):
    NAME = "training"
    IMAGES_PER_GPU = 1  # Reduces training time

    def __init__(self, dataset: utils.ProjectDataset):
        Config.NUM_CLASSES = len(dataset.classes) + 1
        super().__init__()
        self.pr_dataset = dataset


class TrainDataset(mrcnn_utils.Dataset):

    def __init__(self, dataset: utils.ProjectDataset, dataset_dir, class_map=None):
        super().__init__(class_map)
        self.pr_dataset = dataset
        self.dataset_dir = dataset_dir
        amount = len(dataset.images)
        self.eval_items = np.random.choice(amount, int(amount * EVAL_PART), replace=False)

        classes = dataset.classes
        for i in range(len(classes)):
            self.add_class(dataset.source, i, classes[i])

    def bmp_to_binary(self, path):
        # img = Image.open(path)
        # h, w = img.size
        # pixels = list(img.getdata())
        # aux = []
        # for x in range(h):
        #     aux.append(pixels[x * w: x * w + w])
        # for y in range(w):
        #     aux[x].append(pixels[x*h + y])

        # return aux
        # return np.array(img.getdata(),
        #                 np.uint8).reshape(img.size[1], img.size[0], 3)
        return skimage.io.imread(path)

    def add_dataset_image(self, id, image_data):
        self.add_image(self.pr_dataset.source,
                       id,
                       self.pr_dataset.get_image_file(self.dataset_dir, image_data, auto_load=True))

    def load_data_train(self):
        for i in range(len(self.pr_dataset.images)):
            if i not in self.eval_items:
                self.add_dataset_image(i, self.pr_dataset.images[i])

    def load_data_eval(self):
        for i in self.eval_items:
            self.add_dataset_image(i, self.pr_dataset.images[i])

    def load_mask(self, image_id):

        mask_files, class_ids = self.pr_dataset.get_mask_files(self.dataset_dir, self.pr_dataset.images[image_id],
                                                               auto_load=True)
        masks_bytes = []
        for file in mask_files:
            masks_bytes.append(self.bmp_to_binary(file))

        masks = np.stack(masks_bytes, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        return masks, class_ids


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--dataset_dir', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--logs', required=False,
                        default="./log/",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--dataset_config', required=True,
                        metavar="/path/to/json/",
                        help='Path to json file from labelbox')

    args = parser.parse_args()

    dataset_info = utils.ProjectDataset(args.dataset_config)

    config = TrainConfig(dataset=dataset_info)
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    dataset_train = TrainDataset(dataset=dataset_info, dataset_dir=args.dataset_dir)
    dataset_train.load_data_train()
    dataset_train.prepare()

    dataset_eval = TrainDataset(dataset=dataset_info, dataset_dir=args.dataset_dir)
    dataset_eval.load_data_eval()
    dataset_eval.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_eval,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_eval,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_eval,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)
