import tensorflow as tf
import numpy as np

# model: f(x)=c*x

c = 5


def f(x):
    return c * x


# data
x_data = np.arange(-10, 10).reshape((-1, 1))
y_data = f(x_data).reshape((-1, 1))

x = tf.placeholder(tf.float32, shape=[None, 1])
c_var = tf.get_variable('linear_coeff', [1], dtype=tf.float32, initializer=tf.zeros_initializer)
y_pred = c_var * x

# predict
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

res = sess.run(y_pred, feed_dict={x: x_data})
print(res)

# train
y_true = tf.placeholder(tf.float32, shape=[None, 1])
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(100):
    _, loss_value = sess.run((train, loss), feed_dict={x: x_data, y_true: y_data})
    print(loss_value)

print(sess.run(y_pred, feed_dict={x: x_data}))
