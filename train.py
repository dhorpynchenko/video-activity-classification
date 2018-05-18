import skimage.io

from PIL import Image

from mrcnn.config import Config
from mrcnn import model as modellib, utils as mrcnn_utils
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)
import utils
import numpy as np
from functools import reduce
import os

EVAL_PART = 0.1


class TrainConfig(Config):
    NAME = "training"
    IMAGES_PER_GPU = 1  # Reduces training time

    def __init__(self, datasets: list):
        Config.NUM_CLASSES = reduce(lambda n, dataset: n + len(dataset.classes), datasets, 0) + 1
        super().__init__()


class TrainDataset(mrcnn_utils.Dataset):

    @staticmethod
    def get_model_datasets(datasets: list, dataset_dir):

        train_items = []
        eval_items = []

        classes_tuples = []
        dataset_dirs = []

        class_names = set()
        source_names = set()

        class_id = 1  # 0 id is for background
        for dataset in datasets:

            if dataset.source in source_names:
                raise RuntimeError("Multiple json files has same file names " + dataset.source)
            dir = os.path.join(dataset_dir, dataset.source)
            dataset_dirs.append(dir)
            if not os.path.exists(dir):
                os.mkdir(dir)

            amount = len(dataset.images)
            eval = np.random.choice(amount, int(amount * EVAL_PART), replace=False)
            eval_items.append(eval)
            train_items.append(list(filter(lambda x: x not in eval, range(len(dataset.images)))))

            for class_name in dataset.classes:
                if class_name in class_names:
                    raise RuntimeError("Different json has same class names " + class_name)
                classes_tuples.append({"source": dataset.source, "id": class_id, "class_name": class_name})
                class_names.add(class_name)
                class_id += 1

        return TrainDataset(classes_tuples, datasets, train_items, dataset_dirs), \
               TrainDataset(classes_tuples, datasets, eval_items, dataset_dirs)

    def __init__(self, classes: list, datasets: list, datasets_items: list, dataset_dirs: list):
        super().__init__(class_map=None)
        self.pr_datasets = datasets
        self.dataset_dirs = dataset_dirs
        self.dataset_items = datasets_items

        for item in classes:
            self.add_class(item.get("source"), item.get("id"), item.get("class_name"))

        id = 0
        for i in range(len(datasets_items)):
            dataset = datasets[i]
            dataset_items = datasets_items[i]
            dataset_dir = dataset_dirs[i]
            for item in dataset_items:
                self.add_image(dataset.source,
                               id,
                               dataset.get_image_file(dataset_dir, dataset.images[item], auto_load=True))
                id += 1

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

    def load_mask_path(self, image_id):
        dataset_position = 0
        dataset_item = 0

        for i in range(len(self.dataset_items)):
            d = self.dataset_items[i]
            if image_id >= len(d):
                image_id -= len(d)
            else:
                dataset_position = i
                dataset_item = d[image_id]
                break

        dataset = self.pr_datasets[dataset_position]
        dataset_dir = self.dataset_dirs[dataset_position]
        return dataset.get_mask_files(dataset_dir, dataset.images[dataset_item], auto_load=True)

    def load_mask(self, image_id):

        mask_files, class_ids = self.load_mask_path(image_id)

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
                        nargs="*",
                        help='Path to json files from labelbox')

    args = parser.parse_args()

    dataset_info = []
    for item in args.dataset_config:
        dataset_info.append(utils.ProjectDataset(item))

    config = TrainConfig(datasets=dataset_info)
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    dataset_train, dataset_eval = TrainDataset.get_model_datasets(datasets=dataset_info, dataset_dir=args.dataset_dir)
    dataset_train.prepare()
    dataset_eval.prepare()

    for i in dataset_train.image_info:
        id = i.get("id")
        print("For image " + i.get("path") + " masks are " + str(dataset_train.load_mask_path(id)[0]))

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
