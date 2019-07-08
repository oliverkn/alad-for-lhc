import time
import sys

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from dagmm_mod.compression_net import CompressionNet
from dagmm_mod.estimation_net import EstimationNet
from dagmm_mod.gmm import GMM

from os import makedirs
from os.path import exists, join

from core.skeleton import AbstractAnomalyDetector


class DAGMM(AbstractAnomalyDetector):
    """ Deep Autoencoding Gaussian Mixture Model.

    This implementation is based on the paper:
    Bo Zong+ (2018) Deep Autoencoding Gaussian Mixture Model
    for Unsupervised Anomaly Detection, ICLR 2018
    (this is UNOFFICIAL implementation)
    """

    MODEL_FILENAME = "DAGMM_model"
    SCALER_FILENAME = "DAGMM_scaler"

    def __init__(self, config):
        """
        Parameters
        ----------
        comp_hiddens : list of int
            sizes of hidden layers of compression network
            For example, if the sizes are [n1, n2],
            structure of compression network is:
            input_size -> n1 -> n2 -> n1 -> input_sizes
        comp_activation : function
            activation function of compression network
        est_hiddens : list of int
            sizes of hidden layers of estimation network.
            The last element of this list is assigned as n_comp.
            For example, if the sizes are [n1, n2],
            structure of estimation network is:
            input_size -> n1 -> n2 (= n_comp)
        est_activation : function
            activation function of estimation network
        est_dropout_ratio : float (optional)
            dropout ratio of estimation network applied during training
            if 0 or None, dropout is not applied.
        minibatch_size: int (optional)
            mini batch size during training
        epoch_size : int (optional)
            epoch size during training
        learning_rate : float (optional)
            learning rate during training
        lambda1 : float (optional)
            a parameter of loss function (for energy term)
        lambda2 : float (optional)
            a parameter of loss function
            (for sum of diagonal elements of covariance)
        normalize : bool (optional)
            specify whether input data need to be normalized.
            by default, input data is normalized.
        random_seed : int (optional)
            random seed used when fit() is called.
        """
        self.config = config

        self.comp_net = CompressionNet(config.comp_hiddens, config.comp_activation)
        self.est_net = EstimationNet(config.est_hiddens, config.est_activation)
        self.est_dropout_ratio = config.est_dropout_ratio

        n_comp = config.est_hiddens[-1]
        self.gmm = GMM(n_comp)

        self.minibatch_size = config.minibatch_size
        self.epoch_size = config.epoch_size
        self.learning_rate = config.learning_rate
        self.lambda1 = config.lambda1
        self.lambda2 = config.lambda2

        self.normalize = config.normalize
        self.scaler = None
        self.seed = config.random_seed

        self.graph = None
        self.sess = None

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def get_anomaly_scores(self, x):
        return self.predict(x)

    def fit(self, x, logdir, evaluator):
        """ Fit the DAGMM model according to the given data.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data.
        """
        n_samples, n_features = x.shape

        if self.normalize:
            self.scaler = scaler = StandardScaler()
            x = scaler.fit_transform(x)

        with tf.Graph().as_default() as graph:
            self.graph = graph
            tf.set_random_seed(self.seed)
            np.random.seed(seed=self.seed)

            # Create Placeholder
            self.input = input = tf.placeholder(
                dtype=tf.float32, shape=[None, n_features])
            self.drop = drop = tf.placeholder(dtype=tf.float32, shape=[])

            # Build graph
            z, x_dash = self.comp_net.inference(input)
            gamma = self.est_net.inference(z, drop)
            self.gmm.fit(z, gamma)
            energy = self.gmm.energy(z)

            self.energy = energy

            self.x_dash = x_dash

            # Loss function
            loss = (self.comp_net.reconstruction_error(input, x_dash) +
                    self.lambda1 * tf.reduce_mean(energy) +
                    self.lambda2 * self.gmm.cov_diag_loss())

            # Minimizer
            minimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

            # Number of batch
            n_batch = (n_samples - 1) // self.minibatch_size + 1

            # Create tensorflow session and initilize
            init = tf.global_variables_initializer()

            self.sess = tf.Session(graph=graph)
            self.sess.run(init)

            # Training
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)

            step = 0

            for epoch in range(self.epoch_size):
                print('---------- EPOCH %s ----------' % epoch)

                begin = time.time()

                for batch in range(n_batch):
                    i_start = batch * self.minibatch_size
                    i_end = (batch + 1) * self.minibatch_size
                    x_batch = x[idx[i_start:i_end]]

                    self.sess.run(minimizer, feed_dict={
                        input: x_batch, drop: self.est_dropout_ratio})

                    # end of batch
                    step += 1

                    if step % 10_000 == 0:
                        display_progression_epoch(begin, batch, n_batch)

                    if self.config.enable_eval and step % self.config.eval_freq == 0:
                        print('evaluating at step %s' % step)
                        evaluator.evaluate(self, step, {})
                        evaluator.save_results(logdir)

                # end of epoch
                print("Epoch %d | time = %ds" % (epoch, time.time() - begin))

            # Fix GMM parameter
            fix = self.gmm.fix_op()
            self.sess.run(fix, feed_dict={input: x, drop: 0})
            self.energy = self.gmm.energy(z)

            tf.add_to_collection("save", self.input)
            tf.add_to_collection("save", self.energy)

            self.saver = tf.train.Saver()

    def predict(self, x):
        """ Calculate anormaly scores (sample energy) on samples in X.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Data for which anomaly scores are calculated.
            n_features must be equal to n_features of the fitted data.

        Returns
        -------
        energies : array-like, shape (n_samples)
            Calculated sample energies.
        """
        if self.sess is None:
            raise Exception("Trained model does not exist.")

        if self.normalize:
            x = self.scaler.transform(x)

        energies = self.sess.run(self.energy, feed_dict={self.input: x})
        return energies

    def save(self, fdir):
        """ Save trained model to designated directory.
        This method have to be called after training.
        (If not, throw an exception)

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
            If not exists, it is created automatically.
        """
        if self.sess is None:
            raise Exception("Trained model does not exist.")

        if not exists(fdir):
            makedirs(fdir)

        model_path = join(fdir, self.MODEL_FILENAME)
        self.saver.save(self.sess, model_path)

        if self.normalize:
            scaler_path = join(fdir, self.SCALER_FILENAME)
            joblib.dump(self.scaler, scaler_path)

    def load(self, fdir):
        """ Restore trained model from designated directory.

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
        """
        if not exists(fdir):
            raise Exception("Model directory does not exist.")

        model_path = join(fdir, self.MODEL_FILENAME)
        meta_path = model_path + ".meta"

        with tf.Graph().as_default() as graph:
            self.graph = graph
            self.sess = tf.Session(graph=graph)
            self.saver = tf.train.import_meta_graph(meta_path)
            self.saver.restore(self.sess, model_path)

            self.input, self.energy = tf.get_collection("save")

        if self.normalize:
            scaler_path = join(fdir, self.SCALER_FILENAME)
            self.scaler = joblib.load(scaler_path)


def display_progression_epoch(start_time, j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write('time: %s sec | progression: %s / %s (%s %%)' %
                     (int(time.time() - start_time), j, id_max, batch_progression) + chr(13))
    _ = sys.stdout.flush
