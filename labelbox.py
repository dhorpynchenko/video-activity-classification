import json
import os
import shutil
import urllib.request

LABEL = "Label"
MASKS = "Masks"
SKIP = "Skip"
LABELED_DATA = "Labeled Data"
ERROR = "error"

bad_images = {"vpkitefly":[16, 35, 135, 152, 153, 154, 155, 159, 183, 209, 228, 253, 255, 307, 413, 443, 465, 509, 535,
                           547, 594, 601, 632, 637, 652, 654, 667, 376, 606, 554]}


class ImageData:

    def __init__(self, image_id, json_position, class_id):
        self.class_id = class_id
        self.json_position = json_position
        self.image_id = image_id


def loadFile(url, destination):
    print("Loading file " + url)
    file, message = urllib.request.urlretrieve(url)
    shutil.copy(file, destination)
    os.remove(file)


def load_if_absent(url, file):
    if not os.path.exists(file):
        loadFile(url, file)


class ProjectDataset:

    def check_url(self, url: str):
        return url is not None and len(url) > 0 and not url.startswith(ERROR)

    def check_masks_urls(self, item_masks):
        for mask in item_masks.keys():
            if not self.check_url(item_masks[mask]):
                return False
        return True

    def __init__(self, json_file) -> None:
        super().__init__()
        self.source = os.path.splitext(os.path.basename(json_file))[0]
        self.json_file = json.load(open(json_file))
        self.images = []

        classes = set()
        skip_images = bad_images.get(self.source, [])
        for i in range(len(self.json_file)):

            item = self.json_file[i]
            item_labels = item[LABEL]
            item_masks = item.get(MASKS, None)
            # Check consistency
            if item_labels == SKIP \
                    or item_masks is None \
                    or not self.check_url(item[LABELED_DATA]) \
                    or not self.check_masks_urls(item_masks)\
                    or i in skip_images:
                continue

            self.images.append(i)
            classes.update(item_labels.keys())
        self.classes = sorted(classes)

    def get_image_name(self, json_position):
        return "image" + str(json_position)

    def get_mask_name(self, json_position, class_id):
        return self.get_image_name(json_position) + class_id

    def get_image_file(self, directory, id, auto_load=False):
        json_item = self.json_file[id]
        image_name = os.path.join(directory, self.get_image_name(id))
        if auto_load:
            load_if_absent(json_item[LABELED_DATA], image_name)
        return image_name

    def get_mask_files(self, directory, id, auto_load=False):
        json_item = self.json_file[id]

        class_ids = []
        masks = []
        for mask in json_item[MASKS].keys():
            class_ids.append(self.classes.index(mask))
            mask_file = os.path.join(directory, self.get_mask_name(id, mask))
            masks.append(mask_file)
            if auto_load:
                load_if_absent(json_item[MASKS][mask], mask_file)

        return masks, class_ids

    def load_dataset(self, directory):
        for image in self.images:
            self.load_dataset_item(directory, image)

    def load_dataset_item(self, directory, json_position):
        self.get_image_file(directory, json_position, auto_load=True)
        self.get_mask_files(directory, json_position, auto_load=True)
