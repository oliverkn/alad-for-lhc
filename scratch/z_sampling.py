import tensorflow as tf
import numpy as np

x_inp = tf.placeholder(tf.float32, shape=[None, 21], name="input_x")
init_kernel = tf.contrib.layers.xavier_initializer()

with tf.variable_scope('layer1'):
    net = tf.layers.dense(x_inp,
                          units=64,
                          kernel_initializer=init_kernel,
                          name='fc')

with tf.variable_scope('layeru'):
    loc = tf.layers.dense(net,
                          units=4,
                          kernel_initializer=init_kernel,
                          name='fc')

with tf.variable_scope('layers'):
    var = tf.layers.dense(net,
                          units=4,
                          kernel_initializer=init_kernel,
                          name='fc')
    var = tf.nn.relu(var)
dist = tf.distributions.Normal(loc=loc, scale=tf.sqrt(var))
z_out = dist.sample()

data = np.random.uniform(-1, 1, size=(50, 21))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    z, loc_out, var_out = sess.run([z_out, loc, var], feed_dict={x_inp: data})
    print(z.shape)
    print(z)
