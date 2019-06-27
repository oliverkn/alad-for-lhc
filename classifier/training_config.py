import numpy as np

from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras.regularizers import *

# --------------------------------AGENT--------------------------------

# load model, weigths, training configuration (loss, optimizer), state of the optimizer, allowing to resume training exactly where you left off
model_file = None

# load weights only, e.g. use a pretrained network
weights_file = None

# --------------------------------DATA--------------------------------
data_path = '/home/oliverkn/pro/data/4_4/train_supervised.npy'

# --------------------------------TRAINER--------------------------------
batch_size = 256
max_epochs = 501

# result
result_path = '/home/oliverkn/pro/results/4_4/mlp_binary'

remote_training = False
remote_address = '35.188.95.78:2223'

# --------------------------------MODEL--------------------------------
learning_rate = 1e-5
optimizer = Adam(lr=learning_rate)
loss = binary_crossentropy


def get_model_architecture(input_shapes):
    l2_lstm_kernel = None
    l2_lstm_bias = None
    l2_lstm_recurrent = None
    l2_dense_kernel = None
    l2_dense_bias = None
    l2_conv_kernel = None
    l2_conv_bias = None

    l2_lstm_kernel = l2(0.00025)
    l2_lstm_bias = l2(0.00025)
    l2_lstm_recurrent = l2(0.00025)
    # l2_dense_kernel = l2(0.001)
    # l2_dense_bias = l2(0.001)
    # l2_conv_kernel = l2(0.00025)
    # l2_conv_bias = l2(0.00025)

    lstm_input_dropout = 0.0
    recurrent_dropout = 0.0
    dense_dropout = 0.3

    x_flat = Input(shape=input_shapes[0], name='input_dense')

    y = Dense(units=64, activation='relu', kernel_regularizer=l2_dense_kernel, bias_regularizer=l2_dense_bias)(x_flat)
    # y = Dropout(rate=dense_dropout)(y)
    y = Dense(units=64, activation='relu', kernel_regularizer=l2_dense_kernel, bias_regularizer=l2_dense_bias)(y)
    # y = Dropout(rate=dense_dropout)(y)
    y = Dense(units=1, activation='sigmoid', kernel_regularizer=l2_dense_kernel, bias_regularizer=l2_dense_bias)(y)

    return [x_flat], [y]
