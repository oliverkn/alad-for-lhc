import os, sys

import tensorflow as tf
import numpy as np
import sklearn


class LinModel:
    def __init__(self):
        self.sess = tf.get_default_session()

    def init_model(self):
        self.x_in = tf.placeholder(name="input", shape=[None, 1], dtype=tf.float32)
        self.c_var = tf.get_variable(name='linear_coeff', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer)
        self.y_out = self.c_var * self.x_in

        # train
        self.y_true = tf.placeholder(tf.float32, shape=[None, 1])
        self.loss = tf.losses.mean_squared_error(labels=self.y_true, predictions=self.y_out)
        self.optimizer = tf.train.GradientDescentOptimizer(0.0001)
        self.train_op = self.optimizer.minimize(self.loss)

    def pred(self, x):
        return self.sess.run(self.y_out, feed_dict={self.x_in: x})

    def fit(self, x, y, n, batch_size, logdir):
        # init
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # tensor board
        tf.summary.scalar("loss", self.loss)
        sum_op = tf.summary.merge_all()

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

        # train loop
        for epoch in range(n):
            print('prep data')
            n_batch = int(x.shape[0] / batch_size)
            x_feed, y_feed = sklearn.utils.shuffle(x, y)

            cum_loss = 0

            print('feed batches')
            for b in range(n_batch - 1):
                l = b * batch_size
                r = l + batch_size
                _, loss_value, summary = self.sess.run((self.train_op, self.loss, sum_op),
                                                       feed_dict={self.x_in: x_feed[l:r], self.y_true: y_feed[l:r]})
                cum_loss += loss_value
                summary_writer.add_summary(summary)

            print("loss=%s, c=%s" % (cum_loss, sess.run(self.c_var)))

            summary_cum = tf.Summary()
            summary_cum.value.add(tag="cum_loss", simple_value=cum_loss)
            summary_writer.add_summary(summary_cum, global_step=epoch)

            # end of epoch
            # self.end_of_epoch(epoch)

            # callback

            # save
            # self.save(os.path.join(logdir, "%s_model.ckpt" % epoch))
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, '01/model_', global_step=epoch)

            # eval

    def save(self, file):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, file)

    def load(self, file):
        saver = tf.train.Saver()
        saver.restore(self.sess, file)

    def end_of_epoch(self, epoch):
        self.save("%s_model.ckpt" % epoch)


def f(x):
    return 2 * x


x_data = np.linspace(-1, 1, 1000000).reshape((-1, 1))
y_data = f(x_data).reshape((-1, 1))

sess = tf.Session()
with sess.as_default():
    model = LinModel()
    model.init_model()

    model.fit(x_data, y_data, 200, batch_size=256, logdir='01')

    print(model.pred(x_data))

# sess = tf.Session()
# with sess.as_default():
#     model = LinModel()
#     model.init_model()
#
#     model.load("01/model_-0")
#
#     print(model.pred(x_data))
