import datetime
import os
from functools import reduce

import skimage.io
import skimage.color

from PIL import Image
import numpy as np

import labelbox
import utils
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from validation_utils import coordToMatrix, find_centroid, compute_area, find_max_coord, cade_internamente

IMAGES_PER_DATASET = 10


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


    def get_output_matrix_save_filename(source, image_id, mask_index):
        return "log/debug/images/{}_image{}_r_mask{}.bmp".format(source, image_id, mask_index)
    # "mrcnn_class_logits", "mrcnn_bbox_fc",
    # "mrcnn_bbox", "mrcnn_mask"])

    def centre_analisi(fig):

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

        # f = open("diocan.txt","w+")
        # f.write(str(maschere))
        # f.close()
        # print(maschere)
        numMasks = 0
        try:
            numMasks = len(maschere[0][0])
            print("Detected %s objects" % numMasks)
        except Exception as e:
            print(e)
            return 0

        maskRet = []
        # for i in range(numMasks):
        #     img = np.zeros([h, w], dtype=np.bool)
        #     maskRet.append(img)
        for d in range(len(maschere[0][0])):
            maskRet.append([])
            for r in range(len(maschere)):
                maskRet[d].append([])
                for c in range(len(maschere[0])):
                    maskRet[d][r].append((0, 0, 0))
        for r in range(len(maschere)):
            for c in range(len(maschere[0])):
                for h in range(len(maschere[0][0])):
                    if maschere[r][c][h]:
                        maskRet[h][r][c] = (255, 255, 255)

        '''
        for c in range(3):
        img[:, :, c] = np.where(indice == 1, 255, img[:, :, c])

        '''
        centroidi_ret = []
        aree = []
        images = []
        for maskSingle in range(len(maskRet)):
            image = Image.fromarray(np.asarray(maskRet[maskSingle], np.uint8), 'RGB')
            images.append(image)
            # image.show("From model")
            aree.append(compute_area(image))
            ret = find_centroid(image)
            centroidi_ret.append(ret)
        return centroidi_ret, ids, aree, images


    total = 0
    correct = 0
    results = dict()

    for dataset in dataset_info:

        images_to_validate = np.random.choice(dataset.images, IMAGES_PER_DATASET)
        ds_dir = os.path.join(args.dataset_dir, dataset.source)

        for image_id in images_to_validate:

            image_path = dataset.get_image_file(ds_dir, image_id, True)
            print("Processing image %s" % image_path)
            try:
                im = Image.open(image_path)
            except:
                continue
            w, h = im.size

            image_masks = dataset.get_mask_coordinates(image_id)
            masks, _ = dataset.get_mask_files(ds_dir, image_id, True)
            # for item in masks:
            #     Image.open(item).show("Original")

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

            centroidi_lista_mask, idss_mask, aree_mask, images = centre_analisi(image_path)

            for i in range(len(images)):
                images[i].save(get_output_matrix_save_filename(dataset.source, image_id, i))

            for indice in range(len(idss)):

                idss_indice = idss[indice]
                class_results = results.get(classes_ids[idss_indice], None)
                if class_results is None:
                    class_results = [0, 0]
                    results[classes_ids[idss_indice]] = class_results

                class_results[0] += 1
                total += 1

                with open("log/debug/{}_image{}_o_mask{}.txt".format(dataset.source, image_id, indice), 'w') as f:
                    f.write("Image:\n{}\n".format(dataset.get_image_cloud_url(image_id)))
                    for i in range(len(masks)):
                        f.write("Original mask{}\n{}\n".format(i, masks[i]))
                    for i in range(len(images)):
                        f.write("Recognized mask {}\n {}\n".format(i, get_output_matrix_save_filename(dataset.source, image_id, i)))
                    f.write("Original mask:\n{}\n".format(dataset.get_image_cloud_url(image_id)))
                    f.write("id_orig {}\n".format(idss[indice]))
                    f.write("ids_masks {} \n\n".format(idss_mask))
                    f.write("centr_orig {} \n".format(str(centroidi_lista[indice])))
                    f.write("centr_models {}\n".format(str(str(centroidi_lista_mask))))
                    f.write("max {} \n\n".format(max_coord))
                    f.write("aree_orig {} \n".format(aree[indice]))
                    f.write("aree_masks {} \n".format(aree_mask))

                    for indice_mask in range(len(idss_mask)):
                        aree_indice = aree[indice]
                        aree_mask_indice = aree_mask[indice_mask]
                        idss_mask_indice = idss_mask[indice_mask]
                        if idss_mask_indice == idss_indice:
                            if (aree_indice * 0.5) < aree_mask_indice < (aree_indice * 1.5):
                                if cade_internamente(max_coord[indice], centroidi_lista_mask[indice_mask]):
                                    class_results[1] += 1
                                    correct += 1
                                    f.write("Mask {} accepted!\n".format(indice_mask))
                                else:
                                    f.write("Mask {} rejected because of COOR\n".format(indice_mask))
                            else:
                                f.write("Mask {} rejected because of AREA\n".format(indice_mask))
                        else:
                            f.write("Mask {} rejected because of ID\n".format(indice_mask))

    accuracy = correct / total

    with open("log/validate_{:%Y%m%dT%H%M}.txt".format(datetime.datetime.now()), "w") as f:
        for domain in results.keys():
            t = results[domain][0]
            c = results[domain][1]
            f.write("{}: total {}, correct {}, accuracy {}\n".format(domain, t, c, str(c / t)))
        f.write("\n\nTotal: total {}, correct {}, accuracy {}".format(total, correct, accuracy))

    print("Numero di successi: " + str(correct))
    print("Numero totale label: " + str(total))
    print("Percentuale di successo: %s" % accuracy)
