from keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Bidirectional, LSTM, Dense, Activation
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import keras.layers


class ModelConfig:
    SEQUENCE_LENGTH = 25
    FRAME_SIZE = 224


class FrameFeaturesExtractor:
    """

    VGG -> Additional features concatenations -> TFRecord file

    Input - images with masks applied, classes ids
    Output - TFRecord files with single feature embedding vector per frame
    """

    def __init__(self) -> None:
        self.extractor = VGG16(weights='imagenet', include_top=False)

    def extract_features(self, image, size_tuple=(224, 224)):
        # img = image.load_img(img_path, target_size=size_tuple)
        # x = image.img_to_array(img)
        x = np.expand_dims(image, axis=0)
        x = preprocess_input(x)

        features = self.extractor.predict(x)
        features = features.reshape(features.shape[0], -1)
        return features


class RNNModel:
    """
    BiLSTM -> FCL -> Softmax layer -> Activity id
    """

    def __init__(self, is_training=False) -> None:
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
        # self.model.add(Bidirectional(LSTM(10)))
        self.model.add(Dense(5))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def classify(self, frames):
        # classes = self.model.predict(x_test, batch_size=128)
        return 0

    def train(self, dataset_folder):
        pass
