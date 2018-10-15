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
from pandas import Series, DataFrame
import datetime
import json  
import urllib


zhibiao = pd.read_csv('F:/work/tianhe4location/result.csv')

#转换日期格式便于筛选节假日和双休日
d_year = zhibiao['year']
d_month = zhibiao['month']
d_day = zhibiao['day']
date_len = len(d_year)
date_haha = range(date_len)
date_list1 = list(map(lambda i :str(d_year[i])+ '-' + str(d_month[i]) + '-'
                     + str(d_day[i]) , date_haha))
date_list2 = list(map(lambda i :str(d_year[i])+ str(d_month[i])
                     + str(d_day[i]) , date_haha))

def str2time(list_in):
    date_list_new = list()
    for time_id in list_in:
        date_new = datetime.datetime.strptime(time_id,'%Y-%m-%d')
        date_new = date_new.strftime('%Y-%m-%d %H:%M:%S.000')
        date_list_new.append(date_new)
    return date_list_new

date_list_need = str2time(date_list1)

zhibiao['date'] = date_list1
zhibiao['date_new'] = date_list_need
zhibiao.index = zhibiao['date_new']


#计算节假日进行划分
def holiday(date_list):
    server_url = "http://www.easybots.cn/api/holiday.php?d="  
    is_holiday = list()
    for date_id in date_list:
        vop_response = urllib.request.urlopen(server_url+'20180420')  
        vop_data= json.loads(vop_response.read())
        is_holiday.append(vop_data[data])
    return is_holiday
        
is_holiday = holiday(date_list2)


#分别计算两个标准值
def data_devide(grouped_static, day_div):
    workday_zhibiao = list()
    weekend_zhibiao = list()
    for grouped in grouped_static:
        workday_standard.append(grouped[day_div[0]])
        weekend_standard.append(grouped[day_div[1]])
    return workday_standard, weekend_standard

#timestamp 转换为字符串listl
def time_stamp_to_list(time_stamp):
    timelist = []
    for stamp in time_stamp:
        timelist.append(stamp.strftime('%Y-%m-%d %H:%M:%S.000'))
    return timelist

#取周末和工作日数据
def Bday_Weekend(date_start, date_end):
    time_columns = pd.date_range(date_start,  date_end, freq='h')
    #计算工作时间
    time_columns_BD = pd.date_range(date_start,  date_end, freq='B')
    time_bd = time_stamp_to_list(time_columns_BD)
    #计算周末时间
    time_columns_weekend = time_columns.difference(time_columns_BD)
    time_weekend = time_stamp_to_list(time_columns_weekend)
    #返回两组时间
    time_div = list([time_bd, time_weekend])
    return time_div

day_div = Bday_Weekend('2018-4-1', '2018-12-31')
workday_num = len(day_div[0])
weekend_num = len(day_div[1])

#分别计算工作日和非工作日的统计指标

workday, weekend = data_devide(grouped_static_mask, day_div) 


data1 = pd.read_csv('F:/work/tianhe4location.csv')
#data1_cell = list(set(list(data1['小区名称'])))

zhibiao_dropNa = data1[['开始时间', '小区名称', 'MR-RRC连接建立最大用户数', 'YY-RRC连接建立成功率分母', 'YY-E-RAB连接建立成功率分母',
              'YY-切换成功率分母', 'MR-总流量（KB）', '所属网格']]
zhibiao_dropNa.columns = ['time', 'cellname', 'ue', 'rrc', 'erab', 'handover', 'flow', 'net_num']
#data.columns = ['time', 'eNodeB名称', 'cellname', 'flow', 'UE']
#data = data.drop('eNodeB名称', axis=1)

#cell_net = pd.read_csv('F:/work/cell_200net.csv')
#cell_net_need = cell_net[['Description', 'CellName']]
#cell_net_need.columns = ['net_num', 'cellname']

#zhibiao_merge = pd.merge(data, cell_net_need, on='cellname', how='outer')  
#zhibiao_dropNa = zhibiao_merge.dropna()

#zhibiao_dropNa.columns = ['haha', 'time', 'cellname', 'UE', 'qiehuanlv', 'liuliang',
#       'rrcfenzhi', 'rrcfenmu', 'erabfenzi', 'erabfenmu', 'qiehuanfenzhi',
#       'qiehuanfenmu', 'PRB_down', 'PRB_down_used', 'net_num']
#zhibiao_dropNa.drop('haha', axis=1)

zhibiao_date = zhibiao_dropNa['time']
time_list = list()
day_list = list()
month_list = list()
for time_ind in zhibiao_date:
    time_list.append(time_ind[11:13])
    day_list.append(time_ind[8:10])
    month_list.append(time_ind[5:7])
    #zhibiao_dropNa.loc[zhibiao_index, 'hour'] = time_list
zhibiao_dropNa['hour'] = time_list
zhibiao_dropNa['day'] = day_list
zhibiao_dropNa['month'] = month_list

zhibiao_dropNa.index = zhibiao_dropNa['time']
zhibiao_time = list(set(list(zhibiao_dropNa['time'])))




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



#csv_file_name = 'F:/work/TFTS/data/zhibiao1_ue_test.csv'
#reader = tf.contrib.timeseries.CSVReader(csv_file_name)
#train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=20, window_size=72)
#with tf.Session() as sess:
    #data = reader.read_full()
   # coord = tf.train.Coordinator()
    #tf.train.start_queue_runners(sess=sess, coord=coord)
    #data = sess.run(data)
    #coord.request_stop()

    ar = tf.contrib.timeseries.ARRegressor(
            periodicities=24, input_window_size=48, output_window_size=24,
            num_features=1,
            loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)
    
    ar.train(input_fn=train_input_fn, steps=1000)

    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

    (predictions,) = tuple(ar.predict(
            input_fn=tf.contrib.timeseries.predict_continuation_input_fn(evaluation, steps=24)))
    
    prediction_cell.append(predictions['mean'])
    #plt.figure(figsize=(20, 5))
   # plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), label='origin')
    #plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluation')
    #predictions['times'].reshape(-1), predictions['mean'].reshape(-1), label='prediction')
    #plt.xlabel('Time(hour)')
    #plt.ylabel('netflow(G)')
    #plt.title('UE prediction of net "HA480"')
    #plt.legend(loc=4)
    #plt.savefig('F:/work/TFTS/data/output2/UE_predict_result.jpg')


