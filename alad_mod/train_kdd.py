import os
import shutil
import argparse

import tensorflow as tf

from alad_mod import kdd_config as config
from alad_mod.alad import ALAD

from alad_mod import kdd_dataset

from data.raw_loader import *
from evaluation.basic_evaluator import BasicEvaluator

if __name__ == '__main__':
    print('---------- LOADING DATA ----------')
    x, y = kdd_dataset.get_train()
    x_copy = x.copy()
    x_valid, y_valid = kdd_dataset.get_test()

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
    code_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copytree(code_dir, os.path.join(result_dir, 'python'))

    print('---------- STARTING TRAINING ----------')
    with tf.Session() as sess:
        alad = ALAD(config, sess)
        evaluator = BasicEvaluator(x_valid, y_valid, enable_roc=config.enable_roc)
        alad.fit(x, evaluator=evaluator, max_epoch=config.max_epoch, logdir=result_dir)
