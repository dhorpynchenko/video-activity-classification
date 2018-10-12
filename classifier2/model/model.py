import os
from abc import abstractmethod

import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Activation, Reshape, Dropout, GRU
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical


class ModelConfig:

    @staticmethod
    def from_file(path):
        params = {}
        with open(path, "r") as f:
            for line in f.readlines():
                if line.startswith('#'):
                    continue
                parts = line.split('=')
                params[parts[0].strip()] = parts[1].strip()
        return ModelConfig(**params)

    def __init__(self, batch_size, frame_size, sequence_length, rnn_size) -> None:
        super().__init__()
        self.rnn_size = int(rnn_size)
        self.sequence_length = int(sequence_length)
        self.frame_size = int(frame_size)
        self.batch_size = int(batch_size)

    def save(self, path, comment=None):
        with open(path, "w") as f:
            if comment is not None:
                f.write("# {}\n".format(comment))
            fields = vars(self)
            for name in fields.keys():
                f.write("{}={}\n".format(name, fields[name]))


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
    """
    BiLSTM -> FCL -> Softmax layer -> Activity id
    """

    MODEL_CONFIG_FILENAME = "config.cfg"
    WEIGHTS_DEFAULT_FILENAME = "model_weights"
    WEIGHTS_DEFAULT_ID = "default"

    def restore_from_config(self, config_dir):
        self.model_config = ModelConfig.from_file(os.path.join(config_dir, Model.MODEL_CONFIG_FILENAME))
        self._restore_model(config_dir)

    @abstractmethod
    def _restore_model(self, config_dir):
        pass

    def new_model(self, model_config: ModelConfig, input_embedding_size, output_size):
        self.model_config = model_config
        self._init_new_model(input_embedding_size, output_size)

    @abstractmethod
    def _init_new_model(self, input_embedding_size, output_size):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.model_config = None

    @abstractmethod
    def train(self, x_batch, y_batch):
        pass

    @abstractmethod
    def classify(self, frames):
        pass

    def save(self, config_dir):
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        self.model_config.save(os.path.join(config_dir, Model.MODEL_CONFIG_FILENAME))
        self._save_model(config_dir)

    @abstractmethod
    def save_weights(self, config_dir, id=None) -> str:
        pass

    @abstractmethod
    def _save_model(self, config_dir):
        pass


class RNNKerasModelImpl(Model):
    MODEL_FILENAME = "model.json"

    def _restore_model(self, path):
        with open(os.path.join(path, RNNKerasModelImpl.MODEL_FILENAME)) as f:
            self.model = keras.models.model_from_json(f.read())

        self.model.load_weights(
            os.path.join(path, "{}_{}.h5".format(Model.WEIGHTS_DEFAULT_FILENAME, Model.WEIGHTS_DEFAULT_ID)))
        self._update_output_size()
        keras.utils.print_summary(self.model)

    def _init_new_model(self, input_embedding_size, output_size):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(self.model_config.rnn_size, return_sequences=True),
                                     input_shape=(self.model_config.sequence_length, input_embedding_size)))
        self.model.add(Reshape((-1,)))
        self.model.add(Dense(output_size))
        self.model.add(Activation('softmax'))
        opt = RMSprop(lr=0.001, decay=0.)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt)
        self._update_output_size()
        keras.utils.print_summary(self.model)

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.output_size = None

    def _update_output_size(self):
        self.output_size = self.model.layers[-1].output_shape[-1]

    def get_log_callback(self):
        pass

    def classify(self, frames):
        classes = self.model.predict(frames, batch_size=frames.shape[0])
        return np.argmax(classes, 1)

    def train(self, x_batch, y_batch):
        y_batch = to_categorical(y_batch, self.output_size)
        return self.model.train_on_batch(x_batch, y_batch)

    def _save_model(self, config_dir):
        with open(os.path.join(config_dir, RNNKerasModelImpl.MODEL_FILENAME), "w") as f:
            f.write(self.model.to_json())
        self.save_weights(config_dir, Model.WEIGHTS_DEFAULT_ID)

    def save_weights(self, config_dir, id=None) -> str:
        file_name = Model.WEIGHTS_DEFAULT_FILENAME
        if id is not None:
            file_name = "{}_{}".format(file_name, id)
        self.model.save_weights(os.path.join(config_dir, "{}.h5".format(file_name)))
        return file_name


class RNNTensorflowModelImpl(Model):

    SAVE_MODEL_DIR = "model"

    def _restore_model(self, config_dir):
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        # self.saver.re

    def _init_new_model(self, input_embedding_size, output_size):
        with tf.name_scope("inputs"):
            self.input_x = tf.placeholder(tf.float32,
                                          [None, self.model_config.sequence_length, input_embedding_size])
            self.input_y = tf.placeholder(tf.int32, [None])
            input_y_one_hot = tf.one_hot(self.input_y, output_size)

        with tf.name_scope("rnn"):
            fw_cell = RNNTensorflowModelImpl.lstm_cell(self.model_config.rnn_size)
            bw_cell = RNNTensorflowModelImpl.lstm_cell(self.model_config.rnn_size)

            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.input_x,
                                                                             dtype=tf.float32)

        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.reshape(output, (-1, output.shape[1] * output.shape[2]))
        output = tf.tanh(output)

        with tf.name_scope("nn"):
            w = tf.get_variable("w", [output.shape[1], output_size], dtype=output.dtype)
            b = tf.get_variable("b", [output_size], dtype=output.dtype)

            l = tf.nn.xw_plus_b(output, w, b)

        with tf.name_scope(name="loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=l))

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.0001).minimize(self.loss)

        with tf.name_scope("prediction"):
            self.prediction = tf.cast(tf.argmax(tf.nn.softmax(l, axis=-1), axis=-1), dtype=tf.int32)

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    @staticmethod
    def lstm_cell(size):
        return tf.nn.rnn_cell.LSTMCell(size, activation=tf.sigmoid)

    def __init__(self) -> None:
        super().__init__()

    def train(self, x_batch, y_batch):
        feed_dict = {self.input_x: x_batch, self.input_y: y_batch}

        _, loss = self.session.run([self.optimizer, self.loss], feed_dict)
        return loss

    def classify(self, frames):
        feed_dict = {self.input_x: frames}

        pred = self.session.run(self.prediction, feed_dict)

        return pred

    def _save_model(self, config_dir):
        save_dir = os.path.join(config_dir, RNNTensorflowModelImpl.SAVE_MODEL_DIR)
        tf.saved_model.simple_save(self.session, save_dir, {"x": self.input_x, "y": self.input_y},
                                   {"predict": self.prediction})
        self.save_weights(config_dir, Model.WEIGHTS_DEFAULT_ID)

    def save_weights(self, config_dir, id=None) -> str:
        file_name = Model.WEIGHTS_DEFAULT_FILENAME
        if id is not None:
            file_name = "{}_{}".format(file_name, id)
        self.saver.save(self.session, os.path.join(config_dir, file_name))
        return file_name


class ModelFactory:

    @staticmethod
    def restore_tf_model(config_dir) -> Model:
        model = RNNTensorflowModelImpl()
        model.restore_from_config(config_dir)
        return model

    @staticmethod
    def new_tf_model(model_config: ModelConfig, input_embedding_size, output_size) -> Model:
        model = RNNTensorflowModelImpl()
        model.new_model(model_config, input_embedding_size, output_size)
        return model

    @staticmethod
    def restore_keras_model(config_dir) -> Model:
        model = RNNKerasModelImpl()
        model.restore_from_config(config_dir)
        return model

    @staticmethod
    def new_keras_model(model_config: ModelConfig, input_embedding_size, output_size) -> Model:
        model = RNNKerasModelImpl()
        model.new_model(model_config, input_embedding_size, output_size)
        return model
