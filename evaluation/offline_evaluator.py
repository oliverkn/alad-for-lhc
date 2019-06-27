import os

import numpy as np
import sklearn

from evaluation.basic_evaluator import BasicEvaluator

from classifier.mlp_binary_classifier import MlpBinaryClassificationAgent

ad = MlpBinaryClassificationAgent()
ad.load_model_from_yaml('/home/oliverkn/pro/results/classifier_1_1/1/model.yaml')

data_file = '/home/oliverkn/pro/data/1_1/train_supervised.npy'
model_file = '/home/oliverkn/pro/results/classifier_1_1/1/{}_weights.h5'
result_path = '/home/oliverkn/pro/results/classifier_1_1/1/'

max_samples = 100000
start_epoch, stop_epoch = 0, 1000

print('---------- LOADING DATA ----------')
data = np.load(data_file, allow_pickle=True).item()
x, y = data['x'], data['y']

print('data shape:' + str(x.shape))
print('labels shape:' + str(y.shape))

# shuffle and take subset
if x.shape[0] > max_samples:
    print('shuffle and taking subset')
    x, y = sklearn.utils.shuffle(x, y)  # just in case if dataset is not shuffled before
    x, y = x[:max_samples], y[:max_samples]

print('data shape:' + str(x.shape))
print('labels shape:' + str(y.shape))

# run evaluation

evaluator = BasicEvaluator(x, y)

for epoch in range(start_epoch, stop_epoch):

    file = model_file.format(epoch)
    if not os.path.isfile(file): break

    print('load epoch {}'.format(epoch))
    ad.load(file)

    print('evaluate epoch {}'.format(epoch))
    evaluator.evaluate(ad, epoch, {})

evaluator.save_results(result_path)
