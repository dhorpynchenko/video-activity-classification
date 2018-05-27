import os
import shutil
import urllib.request
import zipfile

import numpy as np
import skimage.io

from mrcnn import utils as mrcnn_utils

try:
    from coco.pycocotools.coco import COCO
    from coco.pycocotools import mask as maskUtils
except Exception:
    try:
        from pycocotools.coco import COCO
        from pycocotools import mask as maskUtils
    except Exception:
        print("Run ./coco/install_locally.sh to install pycocotools")
        raise Exception


def get_source_id_string(source: str, id):
    return "{}.{}".format(source, id)


class SegmentationDataset(mrcnn_utils.Dataset):

    @staticmethod
    def get_model_datasets(own_datasets_configs: list, dataset_dir, eval_fraction=0.1):
        own_train_dataset, own_eval_dataset = OwnDataset.get_model_datasets(own_datasets_configs, dataset_dir,
                                                                            eval_fraction)

        own_eval_dataset.prepare()
        own_train_dataset.prepare()

        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(dataset_dir, "train", auto_download=True)
        dataset_train.load_coco(dataset_dir, "valminusminival", auto_download=True)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(dataset_dir, "minival", auto_download=True)
        dataset_val.prepare()

        return SegmentationDataset(dataset_train, own_train_dataset), SegmentationDataset(dataset_train,
                                                                                          own_train_dataset)

    def __init__(self, coco_dataset: mrcnn_utils.Dataset, own_dataset: mrcnn_utils.Dataset):
        super().__init__(class_map=None)
        self.own_dataset = own_dataset
        self.coco_dataset = coco_dataset

        self.coco_len = len(coco_dataset.image_info)
        self.own_len = len(coco_dataset.image_info)

        # {
        #     "source": source,
        #     "id": class_id,
        #     "name": class_name,
        # }
        for class_i in coco_dataset.class_info:
            if class_i["id"] == 0:
                # Skip BG
                continue
            self.add_class(class_i["source"],
                           coco_dataset.map_source_class_id(get_source_id_string(class_i["source"], class_i["id"])),
                           class_i["name"])

        for class_i in own_dataset.class_info:
            if class_i["id"] == 0:
                # Skip BG
                continue
            self.add_class(class_i["source"],
                           own_dataset.map_source_class_id(get_source_id_string(class_i["source"], class_i["id"])),
                           class_i["name"])

        # image_info = {
        #     "id": image_id,
        #     "source": source,
        #     "path": path,
        # }
        id = 0
        for image_i in coco_dataset.image_info:
            self.add_image(image_i["source"], id, image_i["path"])
            id += 1

        for image_i in own_dataset.image_info:
            self.add_image(image_i["source"], id, image_i["path"])
            id += 1

    def load_mask(self, image_id):
        if image_id >= self.coco_len:
            dataset = self.own_dataset
            image_id = image_id - self.coco_len
        else:
            dataset = self.coco_dataset

        masks, class_ids = dataset.load_mask(image_id)

        image = self.image_info[image_id]
        for i in range(len(class_ids)):
            id = class_ids[i]
            # Handle COCO crowds
            # A crowd box in COCO is a bounding box around several instances
            if id < 0:
                id *= -1
                id = self.map_source_class_id(get_source_id_string(image["source"], id))
                id *= -1
            else:
                id = self.map_source_class_id(get_source_id_string(image["source"], id))

            class_ids[i] = id

        return masks, class_ids


class OwnDataset(mrcnn_utils.Dataset):

    @staticmethod
    def get_model_datasets(datasets: list, dataset_dir, eval_fraction):

        if not os.path.exists(os.path.abspath(dataset_dir)):
            os.makedirs(os.path.abspath(dataset_dir))

        train_items = []
        eval_items = []

        classes_tuples = []
        dataset_dirs = []

        class_names = set()
        source_names = set()


        for dataset in datasets:

            if dataset.source in source_names:
                raise RuntimeError("Multiple json files has same file names " + dataset.source)
            dir = os.path.join(dataset_dir, dataset.source)
            dataset_dirs.append(dir)
            if not os.path.exists(dir):
                os.mkdir(dir)

            amount = len(dataset.images)
            eval = np.random.choice(amount, int(amount * eval_fraction), replace=False)
            eval_items.append(eval)
            train_items.append(list(filter(lambda x: x not in eval, range(len(dataset.images)))))

            for class_id in range(len(dataset.classes)):
                class_name = dataset.classes[class_id]
                if class_name in class_names:
                    raise RuntimeError("Different json has same class names " + class_name)
                classes_tuples.append({"source": dataset.source, "id": class_id, "class_name": class_name})
                class_names.add(class_name)

        return OwnDataset(classes_tuples, datasets, train_items, dataset_dirs), \
               OwnDataset(classes_tuples, datasets, eval_items, dataset_dirs)

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
        mask_files, class_ids = dataset.get_mask_files(dataset_dir, dataset.images[dataset_item], auto_load=True)
        # Map to current dataset ids
        for i in range(len(class_ids)):
            class_ids[i] = self.map_source_class_id(get_source_id_string(dataset.source, class_ids[i]))
        return mask_files, class_ids

    def load_mask(self, image_id):

        mask_files, class_ids = self.load_mask_path(image_id)

        masks_bytes = []
        for file in mask_files:
            masks_bytes.append(self.bmp_to_binary(file))

        masks = np.stack(masks_bytes, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        return masks, class_ids


DEFAULT_DATASET_YEAR = "2014"


class CocoDataset(mrcnn_utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        dataset_dir = os.path.join(dataset_dir, "coco")

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                get_source_id_string("coco", annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
