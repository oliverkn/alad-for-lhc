import tensorflow as tf

# --------------------------------DATA--------------------------------
data_path = '/home/oliverkn/pro/data/hlf_set/'
result_path = '/home/oliverkn/pro/results/4_4/dagmm/'

exclude_features = ['nPhoton', 'LepEta']
balance = True

max_valid_samples = 100_000
max_train_samples = 500_000_000  # inf

# --------------------------------HYPERPARAMETERS--------------------------------
comp_hiddens = [120, 60, 30, 4]
comp_activation = tf.nn.tanh
est_hiddens= [20, 8]
est_activation = tf.nn.tanh
est_dropout_ratio = 0.5

minibatch_size = 1024
learning_rate = 0.0001
lambda1 = 0.1
lambda2 = 0.0001

normalize = True
random_seed = 123

# --------------------------------TRAIN_SETTINGS--------------------------------
epoch_size = 1000

eval_freq = 1_000
checkpoint_freq = 10_000

enable_eval = True
enable_checkpoint_save = True
