# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:15:09 2018

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 11 15:24:25 2018

@author: User
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader





zhibiao = pd.read_csv('F:/work/TFTS/data/zhibiao_liuliang_input.csv')
cell_name = set(list(zhibiao.cellname))
prediction_cell = list()

for cell in cell_name:
y = np.array(zhibiao[zhibiao.cellname =='CL314']['flow'])
x = np.array(range(len(y)))
data = {tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,}

reader = NumpyReader(data)

with tf.Session() as sess:
    full_data = reader.read_full()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #print(sess.run(full_data))
    coord.request_stop()

train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader, batch_size=20, window_size=72)

with tf.Session() as sess:
    batch_data = train_input_fn.create_batch()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #one_batch = sess.run(batch_data[0])
    coord.request_stop()




ar = tf.contrib.timeseries.ARRegressor(periodicities=24, input_window_size=48,
                                       output_window_size=24, num_features=1,
                                       loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

ar.train(input_fn=train_input_fn, steps=500)

evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

(predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(evaluation, steps=24)))

prediction_cell.append(predictions['mean'])

