from abc import abstractmethod

import cv2

import utils
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import numpy as np


class Preprocessing:

    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size

    def _resize_to_required(self, image):
        return cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)

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
            for data in self._extract_frames(vidcap, length, width, height, fps):
                yield data
        finally:
            if vidcap is not None:
                vidcap.release()

    @abstractmethod
    def _extract_frames(self, video, length, width, height, fps):
        pass


class NoPreprocessing(Preprocessing):

    def _extract_frames(self, video, length, width, height, fps):
        success = True
        count = 0
        while success and count * fps < length:
            video.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
            success, image = video.read()
            # print('Read a new frame: ', success)
            if success:
                yield self._resize_to_required(image)
                count = count + 1


class MRCNNPreprocessing(Preprocessing):
    class MRCNNPreprocConfig(Config):
        NAME = "preproc"
        IMAGES_PER_GPU = 1  # 1 reduces training time but gives an error https://github.com/matterport/Mask_RCNN/issues/521
        DETECTION_MIN_CONFIDENCE = 0.6

        def __init__(self, classes_ids):
            Config.NUM_CLASSES = len(classes_ids)
            super().__init__()

    def __init__(self, mrcnn_classes_file, mrcnn_weights, max_frames, output_size) -> None:
        super().__init__(output_size)
        # Mask-RCNN model
        self.max_frames = max_frames
        classes_ids = utils.load_class_ids(mrcnn_classes_file)
        config = MRCNNPreprocessing.MRCNNPreprocConfig(classes_ids)

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

    def _extract_frames(self, video, length, width, height, fps):
        interval = max(1, length // self.max_frames)

        count = 0
        skipping = 0
        while count < self.max_frames:
            success, image = video.read()
            if not success:
                break

            # image = cv2.resize(image, (OUTPUT_SIZE, int(image.shape[1] * OUTPUT_SIZE / image.shape[0])))
            # time = datetime.now()
            image = self._resize_to_required(image)
            ids, maschere = self.process_image_mrcnn(image)

            if skipping > 40:
                break

            if len(ids) == 0:
                skipping += 1
                print("Skipping frame %s" % skipping)
                video.set(cv2.CAP_PROP_POS_FRAMES, (count * interval) + (skipping * int(fps / 2)))
                continue

            skipping = 0
            print("Taking frame %s" % count)
            # Apply mask to original image
            self.apply_masks_to_image(image, maschere)
            yield ids, image
            count += 1
            video.set(cv2.CAP_PROP_POS_FRAMES, (count * interval))
