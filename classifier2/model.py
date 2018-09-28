import os

import keras
import numpy as np
from keras import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Bidirectional, LSTM, Dense, Activation, Reshape
from keras.utils.np_utils import to_categorical


class ModelConfig:
    SEQUENCE_LENGTH = 40
    FRAME_SIZE = 224
    BATCH_SIZE = 64


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

    def __init__(self, embedding_size, classes_count, is_training=False) -> None:
        self.classes_count = classes_count
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(10, return_sequences=True),
                                     input_shape=(ModelConfig.SEQUENCE_LENGTH, embedding_size)))
        self.model.add(Reshape((-1,)))
        self.model.add(Dense(classes_count))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        keras.utils.print_summary(self.model)

    def get_log_callback(self):
        pass

    def classify(self, frames):
        classes = self.model.predict(frames, batch_size=len(frames))
        return np.argmax(classes, 1)

    def train(self, x_batch, y_batch):
        y_batch = to_categorical(y_batch, self.classes_count)
        loss = self.model.train_on_batch(x_batch, y_batch)
        print("Loss %s\n" % str(loss))

    def save(self, dir, filename):
        self.model.save_weights(os.path.join(dir, "{}.h5".format(filename)), True)

    def restore(self, path):
        self.model.load_weights(path)
