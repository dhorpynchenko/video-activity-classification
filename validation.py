import skimage.io
import skimage.color

from PIL import Image
import numpy as np

import labelbox
import utils
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from validation_utils import coordToMatrix, find_centroid, compute_area, find_max_coord, cade_internamente


class ValidationConfig(Config):
    NAME = "training"
    STEPS_PER_EPOCH = 1500
    IMAGES_PER_GPU = 1  # 1 reduces training time but gives an error https://github.com/matterport/Mask_RCNN/issues/521
    DETECTION_MIN_CONFIDENCE = 0.6

    def __init__(self, classes_ids):
        Config.NUM_CLASSES = len(classes_ids)
        super().__init__()


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Validate Mask R-CNN')

    parser.add_argument('--model',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--classes', required=True,
                        metavar="/path/to/class_ids_file/",
                        help='Class ids files after training')
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

    classes_ids = utils.load_class_ids(args.classes)
    reversed_classes_ids = utils.make_reversed_dict(classes_ids)

    dataset_info = []
    for item in args.dataset_config:
        dataset_info.append(labelbox.ProjectDataset(item))

    config = ValidationConfig(classes_ids)

    # Create model object in inference mode.
    model = MaskRCNN(mode="inference", model_dir=args.logs, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(args.model, by_name=True, exclude=[])
    # "mrcnn_class_logits", "mrcnn_bbox_fc",
    # "mrcnn_bbox", "mrcnn_mask"])

    total = 0
    correct = 0


    def centre_analisi(fig, h, w):

        # Load image
        image = skimage.io.imread(fig)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        results = model.detect([image], verbose=0)
        r = results[0]
        ids = r['class_ids']
        maschere = r["masks"]
        numMasks = 0
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
                        maskRet[h][r][c] = (255, 255, 255)

        '''
        for c in range(3):
        img[:, :, c] = np.where(indice == 1, 255, img[:, :, c])
        '''
        centroidi_ret = []
        aree = []
        for maskSingle in range(len(maskRet)):
            image = Image.fromarray(img, 'RGB')
            ww, hh = image.size
            aree.append(compute_area(image))
            ret = find_centroid(image, ww, hh)
            centroidi_ret.append(ret)
        return centroidi_ret, ids, aree

    for dataset in dataset_info:

        images_to_validate = [dataset.images[0]]

        for image_id in images_to_validate:

            image_path = dataset.get_image_file(args.dataset_dir, image_id, True)
            print("Processing image %s" % image_path)
            im = Image.open(image_path)
            w, h = im.size

            image_masks = dataset.get_mask_coordinates(image_id)

            maskMat = []

            idss = []
            centroidi_lista = []
            aree = []
            max_coord = []
            for label in image_masks.keys():

                for label_mask in image_masks.get(label):
                    x_coord = []
                    y_coord = []
                    for coordinates in label_mask:
                        y_coord.append(coordinates[0])
                        x_coord.append(coordinates[1])
                    coord = []
                    for ind in range(len(x_coord)):
                        coord.append(x_coord[ind])
                        coord.append(y_coord[ind])
                    immagine = coordToMatrix(coord, w, h)
                    centroidi_lista.append(find_centroid(immagine))
                    aree.append(compute_area(immagine))
                    idss.append(reversed_classes_ids.get(label))
                    max_coord.append(find_max_coord(x_coord, y_coord))

            centroidi_lista_mask, idss_mask, aree_mask = centre_analisi(image_path, w, h)
            for indice in range(len(idss)):
                total += 1
                for indice_mask in range(len(idss_mask)):
                    if (aree[indice] * 0.5) < aree_mask[indice_mask] and aree_mask[indice_mask] < (aree[indice] * 1.5):
                        if cade_internamente(max_coord[indice], centroidi_lista_mask[indice_mask]):
                            if idss_mask[indice_mask] == idss[indice]:
                                correct += 1

    print("Numero di successi: " + str(correct))
    print("Numero totale label: " + str(total))
    print("Percentuale di successo: " + str(float(correct) / float(total)) + "%")
