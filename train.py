import datetime
import os

import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)

import labelbox
import utils
from datasets import SegmentationDataset, get_source_id_string
from mrcnn import model as modellib
from mrcnn.config import Config

EVAL_PART = 0.1
DEFAULT_MODEL = "mask_rcnn_coco.h5"


class TrainConfig(Config):
    NAME = "training"
    STEPS_PER_EPOCH = 1500
    IMAGES_PER_GPU = 2  # 1 reduces training time but gives an error https://github.com/matterport/Mask_RCNN/issues/521

    def __init__(self, dataset: SegmentationDataset):
        Config.NUM_CLASSES = len(dataset.class_info)
        super().__init__()


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN')
    parser.add_argument('--model',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--is_from_scratch', required=True,
                        default=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='Is training from scratch')
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

    required_classes = utils.load_required_classes("required_classes.txt")

    dataset_info = []
    for item in args.dataset_config:
        dataset_info.append(labelbox.ProjectDataset(item))

    dataset_train, dataset_eval = SegmentationDataset.get_model_datasets(own_datasets_configs=dataset_info,
                                                                         dataset_dir=args.dataset_dir,
                                                                         required_classes=required_classes,
                                                                         eval_fraction=EVAL_PART)
    dataset_train.prepare()
    dataset_eval.prepare()

    config = TrainConfig(dataset=dataset_train)
    config.display()

    class_ids_file = os.path.join(args.logs, "class_ids_{:%Y%m%dT%H%M}.txt".format(datetime.datetime.now()))
    with open(class_ids_file, "w") as f:
        for class_item in dataset_train.class_info:
            f.write("{}\t{}\n".format(
                dataset_train.map_source_class_id(get_source_id_string(class_item["source"], class_item["id"])),
                class_item["name"]))

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    model_path = args.model

    # Load weights
    print("Loading weights %s, is from scratch %s", (model_path, str(args.is_from_scratch)))
    model.load_weights(model_path, by_name=True, exclude=[] if not args.is_from_scratch else [
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    # for i in dataset_train.image_info:
    #     id = i.get("id")
    #     print("For image " + i.get("path") + " masks are " + str(dataset_train.load_mask_path(id)[0]))

    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_eval,
                learning_rate=config.LEARNING_RATE,
                epochs=12,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_eval,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_eval,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=5,
                layers='all',
                augmentation=augmentation)
