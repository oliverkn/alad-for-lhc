import tensorflow as tf
import numpy as np
import sklearn


class LinModel:
    def __init__(self):
        self.sess = tf.get_default_session()

    def init_model(self):
        self.graph_model = tf.Graph()
        with self.graph_model.as_default():
            x_in = tf.placeholder(name="input", shape=[None, 1], dtype=tf.float32)
            c_var = tf.get_variable(name='linear_coeff', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer)
            y_out = c_var * x_in

            y_true = tf.placeholder(tf.float32, shape=[None, 1])
            loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_out)
            optimizer = tf.train.GradientDescentOptimizer(0.0001)
            train_op = optimizer.minimize(loss)

            self.__dict__.update(locals())

    def pred(self, x):
        return self.sess.run(self.y_out, feed_dict={self.x_in: x})

    def fit(self, x, y, n, batch_size):
        graph_feed = tf.Graph()
        with graph_feed.as_default():
            # prep data pipeline
            x_placeholder = tf.placeholder(x.dtype, x.shape)
            y_placeholder = tf.placeholder(y.dtype, y.shape)
            dataset = tf.data.Dataset.from_tensor_slices((x_placeholder, y_placeholder))
            dataset = dataset.batch(batch_size).prefetch(batch_size).shuffle(buffer_size=10000)
            iterator = dataset.make_initializable_iterator()
            x_batch, y_batch = iterator.get_next()

        tt = tf.import_graph_def(self.graph_model, input_map={self.x_in: x_batch, self.y_true: y_batch},
                                 return_elements=[self.train_op])

        # init
        init = tf.global_variables_initializer()
        self.sess.run(init)

        dataset = tf.data.Dataset.from_tensor_slices((x_placeholder, y_placeholder))

        # Create batches of data
        dataset = dataset.batch(batch_size)
        # Prefetch data for faster consumption
        dataset = dataset.prefetch(batch_size)

        dataset = dataset.shuffle(buffer_size=10000)

        iterator = dataset.make_initializable_iterator()

        # train loop
        for epoch in range(n):
            print('prep data')
            sess.run(iterator.initializer, feed_dict={x_placeholder: x, y_placeholder: y})
            x_feed, y_feed = iterator.get_next()

            loss = 0
            print('feed batches')
            while True:
                try:
                    _, loss_value = self.sess.run((self.train_op, self.loss),
                                                  feed_dict={self.x_in: x_feed, self.y_true: y_feed})
                    loss += loss_value
                except tf.errors.OutOfRangeError:
                    break

            print("loss=%s, c=%s" % (loss, sess.run(self.c_var)))


def save(self, file):
    saver = tf.train.Saver()
    save_path = saver.save(sess, file)


def load(self, file):
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), file)


def f(x):
    return 2 * x


x_data = np.linspace(-1, 1, 10000000).reshape((-1, 1))
y_data = f(x_data).reshape((-1, 1))

sess = tf.Session()
with sess.as_default():
    model = LinModel()
    model.init_model()

    model.fit(x_data, y_data, 200, batch_size=256)

    print(model.pred(x_data))

# sess = tf.Session()
# with sess.as_default():
#     model = LinModel()
#     model.init_model()
#
#     model.load("model.ckpt")
#
#     print(model.pred(x_data))
