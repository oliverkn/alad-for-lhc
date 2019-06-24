import os

import tensorflow as tf
import keras
from keras import backend as K

from core import skeleton


class SupervisedNNTrainer(skeleton.AbstractTrainer):
    def __init__(self, x, y, evaluator, config, result_dir):
        self.x = x
        self.y = y
        self.evaluator = evaluator
        self.config = config
        self.result_dir = result_dir

    def train(self, agent):
        callback = Callback(agent, self.evaluator, self.config, self.result_dir)

        # if activated, the whole graph will be executed on the remote machine
        if self.config.remote_training:
            print('remote training')
            print('connecting to server: ', self.config.remote_address)
            # resetting the session is important because a remote session is still alive after program execution
            # so if the tensor shapes change in the next run, an error would be thrown
            sess = tf.Session("grpc://" + self.config.remote_address)
            tf.Session.reset("grpc://" + self.config.remote_address)
            K.set_session(sess)

        # train model
        agent.model.fit(self.x, self.y, batch_size=self.config.batch_size, shuffle=True, epochs=self.config.max_epochs,
                        verbose=2, callbacks=[callback])


class Callback(keras.callbacks.Callback):
    def __init__(self, agent, evaluator, config, result_dir):
        self.agent = agent
        self.evaluator = evaluator
        self.config = config
        self.result_dir = result_dir

    def on_train_begin(self, logs=None):  # create new result directory for each training cycle
        # save initial model and evaluation
        self.on_epoch_end(-1, logs)

    def on_epoch_begin(self, epoch, logs=None):
        epoch = epoch + 1
        print('---------- EPOCH %s ----------' % epoch)

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1

        print('saving model')
        self.agent.save(os.path.join(self.result_dir, str(epoch) + '_model.h5'))

        print('saving weights...')
        self.agent.save_weights(os.path.join(self.result_dir, str(epoch) + '_weights.h5'))

        print('evaluating...')
        self.evaluator.evaluate(self.agent, epoch, logs, self.result_dir)

    def on_batch_begin(self, batch, logs=None):
        pass