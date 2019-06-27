import os
import shutil

import tensorflow as tf

from alad_mod import config
from alad_mod.alad import ALAD

from data.raw_loader import *
from evaluation.basic_evaluator import BasicEvaluator

if __name__ == '__main__':
    print('---------- LOADING DATA ----------')
    x = np.load(config.train_data_file, allow_pickle=True)
    data_valid = np.load(config.valid_data_file, allow_pickle=True).item()
    x_valid, y_valid = data_valid['x'], data_valid['y']

    print('training data shapes:' + str(x.shape))
    print('evaluation data shapes:' + str(x_valid.shape))

    if x.shape[0] > config.max_train_samples:
        print('taking subset for training')
        x = x[:config.max_train_samples]

    if x_valid.shape[0] > config.max_valid_samples:
        print('taking subset for validation')
        x_valid, y_valid = x_valid[:config.max_valid_samples], y_valid[:config.max_valid_samples]

    print('training data shapes:' + str(x.shape))
    print('evaluation data shapes:' + str(x_valid.shape))

    print('---------- CREATING RESULT DIRECTORY ----------')

    # create result directory
    max_dir_num = 0
    for subdir in os.listdir(config.result_path):
        if subdir.isdigit():
            max_dir_num = max([max_dir_num, int(subdir)])
    result_dir = os.path.join(config.result_path, str(max_dir_num + 1))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # copy python files
    shutil.copytree('.', os.path.join(result_dir, 'python'))

    print('---------- STARTING TRAINING ----------')
    with tf.Session() as sess:
        alad = ALAD(config, sess)
        evaluator = BasicEvaluator(x_valid, y_valid)
        alad.fit(x, evaluator=evaluator, max_epoch=config.max_epoch, logdir=result_dir)
