# coding: utf-8
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf


def main(_):
    csv_file_name = 'F:/work/TFTS/data/zhibiao1_ue_test.csv'
    reader = tf.contrib.timeseries.CSVReader(csv_file_name)
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=20, window_size=72)
    with tf.Session() as sess:
        data = reader.read_full()
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        data = sess.run(data)
        coord.request_stop()

    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=24, input_window_size=48, output_window_size=24,
        num_features=1,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

    ar.train(input_fn=train_input_fn, steps=1000)

    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

    (predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=72)))

    plt.figure(figsize=(20, 5))
    plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), label='origin')
    plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluation')
    plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1), label='prediction')
    plt.xlabel('Time(hour)')
    plt.ylabel('netflow(G)')
    plt.title('UE prediction of net "HA480"')
    plt.legend(loc=4)
    plt.savefig('F:/work/TFTS/data/output2/UE_predict_result.jpg')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
