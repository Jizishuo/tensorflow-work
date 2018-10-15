# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:11:02 2018

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

zhibiao = pd.read_csv('F:/work/tianhe4location/GL479.csv')
#cell_name = set(list(zhibiao.cellname))
#prediction_cell = list()
#for cell in cell_name:

    #x = np.array(range(1000))
    #noise = np.random.uniform(-0.2, 0.2, 1000)
    #y = np.sin(np.pi * x / 100) + x / 200. + noise
    #plt.plot(x, y)
    #plt.savefig('timeseries_y.jpg')
    #y = np.array(zhibiao[zhibiao.cellname ==cell]['flow'])
    #x = np.array(range(len(y)))
    zhibiao = pd.read_csv('F:/work/tianhe4location/GL479_result.csv')
    y_zhunbei = zhibiao[['UE', 'erab', 'flow', 'handover', 'rrc']]
    y = numpy.array(y_zhunbei[['UE', 'erab', 'flow', 'handover', 
                                 'rrc']])
      #y = np.transpose(y_t)
    x = np.array(range(len(y)))    
    
    
    #y = np.array(zhibiao['flow'])
    #x = np.array(zhibiao['time'])
    data = {tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,}
    reader = NumpyReader(data)
    
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=10, 
                                                           window_size=48)
    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=48, input_window_size=24, output_window_size=24,
        num_features=5, loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)
    
    ar.train(input_fn=train_input_fn, steps=400)
    
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)
    (predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=24)))
    prediction = predictions['mean']
    pre_out = DataFrame(prediction)
    
    pre_out.to_csv('F:/work/tianhe4location/GL479_pre.csv')
    #prediction_cell.append(predictions['mean'])
    plt.figure(figsize=(15, 5))
    plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), label='origin')
    plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluation')
    plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1), label='prediction')
    plt.xlabel('time_step')
    plt.ylabel('values')
    plt.legend(loc=4)
    #plt.savefig('predict_result.jpg')


#if __name__ == '__main__':
#    tf.logging.set_verbosity(tf.logging.INFO)
#    tf.app.run()
