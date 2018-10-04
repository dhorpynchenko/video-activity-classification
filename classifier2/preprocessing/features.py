from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np


class FrameFeaturesExtractor:
    OUTPUT_SIZE = 7 * 7 * 512
    """

    VGG -> Additional features concatenations -> TFRecord file

    Input - images with masks applied, classes ids
    Output - TFRecord files with single feature embedding vector per frame
    """

    def __init__(self) -> None:
        self.extractor = VGG16(weights='imagenet', include_top=False)

    def extract_features(self, images, ids_detected, size_tuple=(224, 224)):
        # img = image.load_img(img_path, target_size=size_tuple)
        x = img_to_array(images)
        x = np.expand_dims(images, axis=0)
        x = preprocess_input(x)

        features = self.extractor.predict(x)
        features = features.reshape(features.shape[0], -1)
        return features
