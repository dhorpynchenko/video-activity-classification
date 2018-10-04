import os
from abc import abstractmethod

import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Activation, Reshape
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical


class ModelConfig:

    @staticmethod
    def from_file(path):
        params = {}
        with open(path, "r") as f:
            for line in f.readline():
                if line.startswith('#'):
                    continue
                parts = line.split('=')
                params[parts[0].strip()] = params[1].strip()
        return ModelConfig(**params)

    def __init__(self, batch_size, frame_size, sequence_length, rnn_size) -> None:
        super().__init__()
        self.rnn_size = rnn_size
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.batch_size = batch_size

    def save(self, path):
        pass


def fit_with_sequence_length(sequences, sequence_length):
    for i in range(len(sequences)):
        x = sequences[i]
        frames = x.shape[0]
        if frames > sequence_length:
            x = x[0:sequence_length, :]
            sequences[i] = x
        elif frames < sequence_length:
            temp = np.zeros(shape=(sequence_length, x.shape[1]))
            temp[0:frames, :] = x
            sequences[i] = temp


class Model:

    @abstractmethod
    @staticmethod
    def restore_from_config(path):
        pass

    def __init__(self, model_config: ModelConfig, input_embedding_size, output_size, is_training=False) -> None:
        super().__init__()

        self.output_size = output_size
        self.input_embedding_size = input_embedding_size
        self.model_config = model_config

    @abstractmethod
    def train(self, x_batch, y_batch):
        pass

    @abstractmethod
    def classify(self, frames):
        pass

    @abstractmethod
    def save(self, config_dir):
        pass


class RNNKerasModel(Model):
    """
    BiLSTM -> FCL -> Softmax layer -> Activity id
    """

    MODEL_FILENAME = "model.json"
    WEIGHTS_DEFAULT_FILENAME = "model_weights_default"

    @staticmethod
    def restore_from_config(path):
        pass

    def __init__(self, model_config, input_embedding_size, output_size, is_training=False) -> None:
        super().__init__(model_config, input_embedding_size, output_size, is_training)
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(self.model_config.rnn_size, return_sequences=True),
                                     input_shape=(self.model_config.sequence_length, self.input_embedding_size)))
        # self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Reshape((-1,)))
        # self.model.add(Dense(classes_count * 5))
        self.model.add(Dense(self.output_size))
        self.model.add(Activation('softmax'))
        opt = RMSprop(lr=0.001, decay=0.)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt)
        keras.utils.print_summary(self.model)

    def get_log_callback(self):
        pass

    def classify(self, frames):
        classes = self.model.predict(frames, batch_size=frames.shape[0])
        return np.argmax(classes, 1)

    def _check_pad_sequence(self, frames):
        return frames

    def train(self, x_batch, y_batch):
        y_batch = to_categorical(y_batch, self.output_size)
        # x_batch = self._check_pad_sequence(x_batch)
        return self.model.train_on_batch(x_batch, y_batch)
        # print("Loss %s\n" % str(loss))

    def fit(self, generator, steps_per_epoch, epochs):
        def check():
            for x, y in generator:
                fit_with_sequence_length(x, self.model_config.sequence_length)
                y = to_categorical(y, self.output_size)
                yield x, y

        self.model.fit_generator(check(), steps_per_epoch, epochs)

    def save_weights(self, dir, filename=WEIGHTS_DEFAULT_FILENAME):
        self.model.save_weights(os.path.join(dir, "{}.h5".format(filename)), True)

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_model(self, dir):
        with open(os.path.join(dir, RNNKerasModel.MODEL_FILENAME), "w") as f:
            f.write(self.model.to_json())

    def load_model1(self, dir):
        with open(os.path.join(dir, RNNKerasModel.MODEL_FILENAME)) as f:
            self.model = keras.models.model_from_json(f.read())


class RNNTensorflowModel(Model):

    @staticmethod
    def restore_from_config(path):
        return RNNTensorflowModel()

    @staticmethod
    def lstm_cell(size):
        return tf.nn.rnn_cell.LSTMCell(size)

    def __init__(self, model_config, input_embedding_size, output_size, is_training=False) -> None:
        super().__init__(model_config, input_embedding_size, output_size, is_training)

        with tf.name_scope("inputs"):
            self.input_x = tf.placeholder(tf.float32,
                                          [None, self.model_config.sequence_length, self.input_embedding_size])
            self.input_y = tf.placeholder(tf.int32, [None])
            input_y_one_hot = tf.one_hot(self.input_y, self.output_size)

        with tf.name_scope("rnn"):
            fw_cell = RNNTensorflowModel.lstm_cell(self.model_config.rnn_size)
            bw_cell = RNNTensorflowModel.lstm_cell(self.model_config.rnn_size)

            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.input_x,
                                                                             dtype=tf.float32)

        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.reshape(output, (-1, output.shape[1] * output.shape[2]))

        with tf.name_scope("nn"):
            w = tf.get_variable("w", [output.shape[1], self.output_size], dtype=output.dtype)
            b = tf.get_variable("b", [self.output_size], dtype=output.dtype)

            l = tf.nn.xw_plus_b(output, w, b)

        with tf.name_scope(name="loss"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_y_one_hot, logits=l))

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.loss)

        with tf.name_scope("prediction"):
            self.prediction = tf.cast(tf.argmax(tf.nn.softmax(l, axis=-1), axis=-1), dtype=tf.int32)

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self, x_batch, y_batch):
        feed_dict = {self.input_x: x_batch, self.input_y: y_batch}

        _, loss = self.session.run([self.optimizer, self.loss], feed_dict)
        return loss

    def classify(self, frames):
        feed_dict = {self.input_x: frames}

        pred = self.session.run(self.prediction, feed_dict)

        return pred

    def save(self, config_dir):
        pass
