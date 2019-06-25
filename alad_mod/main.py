import os
import shutil

from alad_mod import config
from alad_mod.alad import ALAD

from data.raw_loader import *

if __name__ == '__main__':
    print('---------- LOADING DATA ----------')
    data = np.load(config.data_path, allow_pickle=True).item()
    x_train, y_train, x_eval, y_eval = data['x_train'], data['y_train'], data['x_eval'], data['y_eval']

    print('training data shapes:' + str(x_train.shape))
    print('evaluation data shapes:' + str(x_eval.shape))

    print('---------- INITIALIZE TRAINING ----------')

    # get model architecture and compile it

    alad = ALAD(config)

    # loading model if set
    # if cfg.model_file is not None:
    #     print('loading model: ' + cfg.model_file)
    #     anomaly_detector.load(cfg.model_file)

    # load initial weights if set
    # elif cfg.weights_file is not None:
    #     print('loading weights: ' + cfg.model_file)
    #     anomaly_detector.load_weights(cfg.model_file)

    print('---------- CREATING RESULT DIRECTORY ----------')

    # create result directory
    max_dir_num = 0
    for subdir in os.listdir(config.result_path):
        if subdir.isdigit():
            max_dir_num = max([max_dir_num, int(subdir)])
    result_dir = os.path.join(config.result_path, str(max_dir_num + 1))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print('---------- STARTING TRAINING ----------')

    # instantiate evaluator and trainer
    alad.fit(x_train, max_epoch=config.max_epoch, logdir=result_dir)
