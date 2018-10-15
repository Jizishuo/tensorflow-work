# coding: utf-8
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib as plt
plt.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader
import time
from multiprocessing import  Process
import threading
#from numba import jit
#from multiprocessing import Process

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


zhibiao = pd.read_csv('F:\\work-code\\workday.csv')
#cell_list_need = list(set(zhibiao['net_num']))
cell_list_need = list(set(zhibiao['cellname']))
#去cell-name-list
list1 = cell_list_need[0:6]
#print("list1区域:", list(list1))
#共48个 分6次取
list2 = cell_list_need[6:12]
list3 = cell_list_need[12:18]
list4 = cell_list_need[18:24]
list5 = cell_list_need[24:30]
list6 = cell_list_need[30:36]
list7 = cell_list_need[36:42]
list8 = cell_list_need[42:47]

zhibiao.index = zhibiao['cellname']

zhibiao1 = zhibiao.ix[list1]
#print(zhibiao1)
#分8次取到6个不同cell-name的数据
zhibiao2 = zhibiao.ix[list2]
zhibiao3 = zhibiao.ix[list3]
zhibiao4 = zhibiao.ix[list4]
zhibiao5 = zhibiao.ix[list5]
zhibiao6 = zhibiao.ix[list6]
zhibiao7 = zhibiao.ix[list7]
zhibiao8 = zhibiao.ix[list8]

start = time.time()

#@jit
def tensor_flow(net_list,zhibiao_in):
    predict_result = []
    for net in net_list: 
        y = np.array(zhibiao_in[zhibiao_in.cellname == net].iloc[:, 5:10])
        x = np.array(range(len(y)))
        #print(y)
        #print(x)
        data = {
                tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
                tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
                }
        print("net:", net)
        reader = NumpyReader(data)

        train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
                reader, batch_size=10, window_size=48)
        #periodicities=48周期,input_window_size=24, output_window_size=24输入输出 相加等于window_size=48
        #num_features参数表示在一个时间点上观察到的数的维度 5个一个维度
        ar = tf.contrib.timeseries.ARRegressor(
                periodicities=48, input_window_size=24, output_window_size=24,
                num_features=5,
                loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

        ar.train(input_fn=train_input_fn, steps=500)
        evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
        # keys of evaluation: ['covariance', 'loss'损失值, 'mean'预测值, 'observed', 'start_tuple', 'times'时间mean, 'global_step']
        evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)
        #预测未来24小时
        (predictions,) = tuple(ar.predict(
                input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
                        evaluation, steps=24)))
        prediction_re = predictions['mean']
        #prediction_time = predictions['times']
        predict_result.append(prediction_re)

        plt.figure(figsize=(15, 5))

        plt.plot(data['times'].reshape(-1), data['values'].reshape(-1)[:].tolist()[::5], label='origin-UE',color="red")
        #plt.plot(data['times'].reshape(-1), data['values'].reshape(-1)[:].tolist()[1::5], label='origin-erab', color="yellow")
        plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1)[:].tolist()[::5], label='evaluation')
        plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1)[:].tolist()[::5], label='prediction')
        plt.xlabel('time_step')
        plt.ylabel('values')
        plt.legend(loc=4)
        plt.show()
        plt.savefig('predict_result-for%s.jpg' % net)


        #print(prediction_re)
        '''
        [rows, cols] = prediction_re.shape
        for i in range(rows - 1):
            for j in range(cols - 1):
                print(prediction_re[j, i])
        
        a= prediction_re.reshape(-1)

        print(a[:].tolist())
        print("完成")
        '''
    return predict_result
    #plt.savefig('predict_result.jpg')



p1 = threading.Thread(target=tensor_flow(list1, zhibiao1))
#p2 = threading.Thread(target=tensor_flow(list2, zhibiao2))
#p3 = threading.Thread(target=tensor_flow(list3, zhibiao3))
#p4 = threading.Thread(target=tensor_flow(list4, zhibiao4))
#p5 = threading.Thread(target=tensor_flow(list5, zhibiao5))
#p6 = threading.Thread(target=tensor_flow(list6, zhibiao6))
#p7 = threading.Thread(target=tensor_flow(list7, zhibiao7))
#p8 = threading.Thread(target=tensor_flow(list8, zhibiao8))

p1.start()
#p2.start()
#p3.start()
#p4.start()
#p5.start()
#p6.start()
#p7.start()
#p8.start()

'''
p1.join()
p2.join()
p3.join()
p4.join()
p5.join()
p6.join()
p7.join()
p8.join()
'''
    
end = time.time()
print(end-start)
#print(predict_result)


