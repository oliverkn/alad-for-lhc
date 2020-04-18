import tensorflow as tf
import numpy as np

# --------------------------------DATA--------------------------------
data_file = '/cluster/home/knappo/record_6021/data_hlf.hdf5'
result_path = '/cluster/home/knappo/results/record_6021/'

# preprocessor
cont_list = ['HT', 'mass_jet', 'n_jet', 'n_bjet', 'lep_pt', 'lep_iso_ch', 'lep_iso_neu',
             'lep_iso_gamma', 'MET', 'METo', 'METp', 'MT', 'pt_mu', 'mass_mu', 'pt_ele',
             'mass_ele', 'n_neu', 'n_ch']

disc_list = ['lep_charge', 'n_mu', 'n_ele']

categories = [None] * len(disc_list)
categories[0] = [-1, 1]
categories[1] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
categories[2] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# training set
max_train_samples = 10_000_000

# --------------------------------HYPERPARAMETERS--------------------------------
input_dim = 42
latent_dim = 16

learning_rate = 1e-5
batch_size = 50
init_kernel = tf.contrib.layers.xavier_initializer()
ema_decay = 0.999
do_spectral_norm = True
allow_zz = True
fm_degree = 1

# --------------------------------TRAIN_SETTINGS--------------------------------
weights_file = None

max_epoch = 10000

sm_write_freq = 10_000  # number of batches
eval_freq = 50_000
checkpoint_freq = 500_000

enable_sm = False
enable_eval = False
enable_checkpoint_save = True


# --------------------------------MODELS--------------------------------

def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))
    else:
        return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def encoder(x_inp, is_training=False, getter=None, reuse=False,
            do_spectral_norm=False):
    """ Encoder architecture in tensorflow

    Maps the data into the latent space

    Args:
        x_inp (tensor): input data for the encoder.
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not

    Returns:
        net (tensor): last activation layer of the encoder

    """
    with tf.variable_scope('encoder', reuse=reuse, custom_getter=getter):
        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(x_inp,
                                  units=64,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(x_inp,
                                  units=64,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)

        name_net = 'layer_out'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=latent_dim,
                                  kernel_initializer=init_kernel,
                                  name='fc')

    return net


def decoder(z_inp, is_training=False, getter=None, reuse=False):
    """ Generator architecture in tensorflow

    Generates data from the latent space

    Args:
        z_inp (tensor): input variable in the latent space
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not

    Returns:
        net (tensor): last activation layer of the generator

    """
    with tf.variable_scope('generator', reuse=reuse, custom_getter=getter):
        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(z_inp,
                                  units=64,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=64,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = tf.nn.relu(net)

        name_net = 'layer_out'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=input_dim,
                                  kernel_initializer=init_kernel,
                                  name='fc')

    return net


def discriminator_xz(x_inp, z_inp, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=False):
    """ Discriminator architecture in tensorflow
    Discriminates between pairs (E(x), x) and (z, G(z))
    Args:
        x_inp (tensor): input data for the discriminator.
        z_inp (tensor): input variable in the latent space
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not
    Returns:
        logits (tensor): last activation layer of the discriminator (shape 1)
        intermediate_layer (tensor): intermediate layer for feature matching
    """
    with tf.variable_scope('discriminator_xz', reuse=reuse, custom_getter=getter):
        # D(x)
        name_x = 'x_layer_1'
        with tf.variable_scope(name_x):
            x = tf.layers.dense(x_inp,
                                units=128,
                                kernel_initializer=init_kernel,
                                name='fc')
            x = tf.layers.batch_normalization(x,
                                              training=is_training,
                                              name='batch_normalization')
            x = leakyReLu(x)

        # D(z)
        name_z = 'z_layer_1'
        with tf.variable_scope(name_z):
            z = tf.layers.dense(z_inp, 128, kernel_initializer=init_kernel)
            z = leakyReLu(z)
            z = tf.layers.dropout(z, rate=0.5, name='dropout', training=is_training)

        # D(x,z)
        y = tf.concat([x, z], axis=1)

        name_y = 'y_layer_1'
        with tf.variable_scope(name_y):
            y = tf.layers.dense(y,
                                128,
                                kernel_initializer=init_kernel)
            y = leakyReLu(y)
            y = tf.layers.dropout(y, rate=0.5, name='dropout', training=is_training)

        intermediate_layer = y

        name_y = 'y_layer_2'
        with tf.variable_scope(name_y):
            logits = tf.layers.dense(y,
                                     1,
                                     kernel_initializer=init_kernel)

    return logits, intermediate_layer


def discriminator_xx(x, rec_x, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=False):
    """ Discriminator architecture in tensorflow
    Discriminates between (x,x) and (x,rec_x)
    Args:
        x (tensor): input from the data space
        rec_x (tensor): reconstructed data
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not
    Returns:
        logits (tensor): last activation layer of the discriminator
        intermediate_layer (tensor): intermediate layer for feature matching
    """
    with tf.variable_scope('discriminator_xx', reuse=reuse, custom_getter=getter):
        net = tf.concat([x, rec_x], axis=1)

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=128,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net)
            net = tf.layers.dropout(net, rate=0.2, name='dropout', training=is_training)

        intermediate_layer = net

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            logits = tf.layers.dense(net,
                                     units=1,
                                     kernel_initializer=init_kernel,
                                     name='fc')

    return logits, intermediate_layer


def discriminator_zz(z, rec_z, is_training=False, getter=None, reuse=False,
                     do_spectral_norm=False):
    """ Discriminator architecture in tensorflow
    Discriminates between (z,z) and (z,rec_z)
    Args:
        z (tensor): input from the latent space
        rec_z (tensor): reconstructed data
        is_training (bool): for batch norms and dropouts
        getter: for exponential moving average during inference
        reuse (bool): sharing variables or not
    Returns:
        logits (tensor): last activation layer of the discriminator
        intermediate_layer (tensor): intermediate layer for feature matching
    """
    with tf.variable_scope('discriminator_zz', reuse=reuse, custom_getter=getter):
        net = tf.concat([z, rec_z], axis=-1)

        name_net = 'layer_1'
        with tf.variable_scope(name_net):
            net = tf.layers.dense(net,
                                  units=latent_dim,
                                  kernel_initializer=init_kernel,
                                  name='fc')
            net = leakyReLu(net, 0.2, name='conv2/leaky_relu')
            net = tf.layers.dropout(net, rate=0.2, name='dropout',
                                    training=is_training)

        intermediate_layer = net

        name_net = 'layer_2'
        with tf.variable_scope(name_net):
            logits = tf.layers.dense(net,
                                     units=1,
                                     kernel_initializer=init_kernel,
                                     name='fc')

    return logits, intermediate_layer
