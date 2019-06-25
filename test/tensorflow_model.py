import tensorflow as tf
import numpy as np


class LinModel:
    def __init__(self):
        x_in = tf.placeholder(name="input", shape=[None, 1], dtype=tf.float32)
        c_var = tf.get_variable(name='linear_coeff', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer)
        y_pred = c_var * x_in

        self.__dict__.update(locals())

    def init_model(self):
        init = tf.global_variables_initializer()
        tf.get_default_session().run(init)

    def pred(self, x):
        return tf.get_default_session().run(self.y_pred, feed_dict={self.x_in: x})

    def fit(self, x, y, n):
        y_true = tf.placeholder(tf.float32, shape=[None, 1])
        loss = tf.losses.mean_squared_error(labels=y_true, predictions=self.y_pred)

        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

        for i in range(n):
            _, loss_value = sess.run((train, loss), feed_dict={self.x_in: x, y_true: y})
            print(loss_value)

    def save(self, file):
        saver = tf.train.Saver()
        save_path = saver.save(sess, file)

    def load(self, file):
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), file)


def f(x):
    return 2 * x


x_data = np.arange(-10, 10).reshape((-1, 1))
y_data = f(x_data).reshape((-1, 1))

# sess = tf.Session()
# with sess.as_default():
#     model = LinModel()
#     model.init_model()
#     print(model.pred(x_data))
#
#     # `sess.graph` provides access to the graph used in a `tf.Session`.
#     writer = tf.summary.FileWriter(".", sess.graph)
#
#     model.fit(x_data, y_data, 20)
#     print(model.pred(x_data))
#
#     model.save("model.ckpt")

sess = tf.Session()
with sess.as_default():
    model = LinModel()
    model.init_model()

    model.load("model.ckpt")

    print(model.pred(x_data))
