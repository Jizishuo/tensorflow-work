# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:38:16 2018

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

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from pandas import DataFrame

#from os import path
 
import numpy 
from pandas import DataFrame

import matplotlib
matplotlib.use("agg")

import datetime
import os
#import sys  

#当前文件的路径
pwd = os.getcwd()
#当前文件的父路径
father_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+".") + os.path.sep

predicted_path = father_path +'output' +os.path.sep 
predicted_picture_path = father_path +'output' +os.path.sep + 'picture' + os.path.sep

data_basic_path = father_path +'basic_data' +os.path.sep 


#import json  
#import urllib

#def holiday(date_list):
    #server_url = "http://www.easybots.cn/api/holiday.php?d="  
    #vop_response = urllib.request.urlopen(server_url + date_list)  
    #vop_data= json.loads(vop_response.read())
    #is_holiday = vop_data[date_list]
    #return is_holiday

def str2time_1(list_in):
    date_list_new = list()
    for time_id in list_in:
        date_new = datetime.datetime.strptime(time_id,'%Y/%m/%d/%H')
        date_new = date_new.strftime('%Y-%m-%d %H:%M:%S.000')
        date_list_new.append(date_new)
    return date_list_new

def date_list_1(zhibiao):
    d_year = zhibiao['year']
    d_month = zhibiao['month']
    d_day = zhibiao['day']
    d_hour = zhibiao['hour']
    date_len = len(d_year)
    date_haha = range(date_len)
    date_list1 = list(map(lambda i :str(d_year[i])+ '/' + str(d_month[i]) + '/'
                     + str(d_day[i])+ '/' + str(d_hour[i]) , date_haha))
    #date_list2 = list(map(lambda i :str(d_year[i])+ str(d_month[i])
                     #+ str(d_day[i]) , date_haha))

    date_list_out = str2time_1(date_list1)
    return date_list_out




def str2time(list_in):
    date_list_new = list()
    for time_id in list_in:
        date_new = datetime.datetime.strptime(time_id,'%Y/%m/%d')
        date_new = date_new.strftime('%Y-%m-%d %H:%M:%S.000')
        date_list_new.append(date_new)
    return date_list_new

def date_list(zhibiao):
    d_year = zhibiao['year']
    d_month = zhibiao['month']
    d_day = zhibiao['day']
    date_len = len(d_year)
    date_haha = range(date_len)
    date_list1 = list(map(lambda i :str(d_year[i])+ '/' + str(d_month[i]) + '/'
                     + str(d_day[i]) , date_haha))
    #date_list2 = list(map(lambda i :str(d_year[i])+ str(d_month[i])
                     #+ str(d_day[i]) , date_haha))

    date_list_out = str2time(date_list1)
    return date_list_out
  
def isworkday(data):
    date_thisday = data[0]
    if date_thisday in workday_2018:
        return "1"
    else:
        return "0"
    
    
def ratio_mark(data_input, standard):
    data_input_list = DataFrame(data_input, columns=['flow'])
    mark_1 = abs(data_input_list.where(data_input_list['flow']<standard)/standard)
    mark_out = mark_1.fillna(1)
    return mark_out
    
    


def not_ok(data_input, lastday_input, data_input_ratio, net_item, flow_standard, alert_standard):
    
    ratio_need = data_input_ratio[['UE', 'erab', 'handover', 'rrc']]
    ratio_need_columns = ratio_need.columns
    ratio_out = DataFrame()
    for columns_need in ratio_need_columns:
        ratio_item = ratio_need[columns_need]
        ratio_isok = list(map(lambda y:0  if y>alert_standard else 1, ratio_item))
        ratio_out[columns_need] = ratio_isok
        
#    rows_input = ratio_need.iloc[:,0].size
#    columns_input = ratio_need.columns.size
#    for i in range(rows_input):
#        for j in range(columns_input):
#            value_need = ratio_need.iloc[[i]].values[0][j]
#            if value_need > 0.3 :
#                isok = 0
#            else:
#                isok = 1
#            ratio_need.iloc[[i]].values[0][j] = isok
                
    flow_ratio = DataFrame(data_input_ratio['flow'],columns=['flow'])
    flow_ratio.index = range(len(flow_ratio))
    flow_data_pred = data_input['flow']    
    flow_data_pred.index = range(len(flow_data_pred))  
    last_flow = lastday_input['flow']
    
    
    flow_mark = ratio_mark(flow_data_pred, flow_standard)# 超过3G则不考虑比例转换
    flow_ratio_out = DataFrame(np.array(flow_ratio)*np.array(flow_mark), columns=['flow'])
    flow_ratio_out_series = flow_ratio_out['flow']
    
    flow_isok = list(map(lambda y:0  if y>alert_standard else 1, flow_ratio_out_series))
    
    ratio_out['flow'] = flow_isok
    
    #画图
    notok_hours = 24 - ratio_out.sum(axis=0)
    if notok_hours['flow'] > 3:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.plot(range(24), flow_data_pred, label="flow_pred", color="g")
        ax.plot(range(24), last_flow, label="flow_lastday", color="r")
        ax.set_ylabel('GB')
        ax.set_xlabel('Hour')
        plt.legend(loc="upper left")
        plt.title('net_num: ' + net_item + '  flow lastday')
        #plt.show()
        plt.savefig(predicted_picture_path + net_item +'.jpg')    
    
    return ratio_out

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  #计算当天指标是否有异常
  predicted_zhibiao = pd.read_csv(predicted_path + 'predicted_last.csv')
  predicted_zhibiao.columns = ['UE', 'erab', 'flow', 'handover', 'rrc', 'net_num']
  zhibiao_lastday = pd.read_csv(father_path + 'today.csv')
  
  zhibiao_last_day = zhibiao_lastday[['UE' , 'erab', 'flow', 'handover', 'rrc', 'net_num']] 
  net_num_list = list(set(list(predicted_zhibiao['net_num'])))
  
  bijiao_result = DataFrame()
  
  for net_item in net_num_list:
      pred_zhibiao = predicted_zhibiao[predicted_zhibiao.net_num == net_item][
              ['UE' , 'erab', 'flow', 'handover', 'rrc']]
      last_zhibiao = zhibiao_last_day[zhibiao_last_day.net_num == net_item][[
              'UE' , 'erab', 'flow', 'handover', 'rrc']]
      #bijiao = DataFrame(np.array(pred_zhibiao) - np.array(last_zhibiao), 
                         #columns=['hour', 'UE' , 'erab', 'flow', 'handover', 'rrc'])
      bijiao = np.array(pred_zhibiao) - np.array(last_zhibiao)                                         
      bijiao_ratio = np.array(bijiao)/np.array(pred_zhibiao)
      
      bijiao_da = DataFrame(bijiao, columns=['UE' , 'erab', 
                         'flow', 'handover', 'rrc'])
      bijiao_ratio_da = DataFrame(bijiao_ratio, columns=['UE' , 'erab', 
                   'flow', 'handover', 'rrc'])
           
      bijiao_notok = not_ok(pred_zhibiao, last_zhibiao, bijiao_ratio_da, net_item, 20, 0.5) # 3表示比例转换的流量基准， 0.5表示告警门限 
      bijiao_notok['net_num'] = net_item
      bijiao_result = bijiao_result.append(bijiao_notok) 
      
  #bijiao_result.to_csv("F:/work/tianhe4location/output/isok.csv")   
  
  
  #计算是否为公休日
  workday_calender = pd.read_csv(data_basic_path + '2018workday.csv')
  workday_2018 = str2time(list(workday_calender['date']))   
  zhibiao_oneday = pd.read_csv(father_path + 'today.csv')
  data = date_list(zhibiao_oneday)
  is_workday = isworkday(data)
  
  if is_workday:
      zhibiao = pd.read_csv(father_path + 'workday.csv')
      zhibiao1 = zhibiao.append(zhibiao_oneday)
      my_file = father_path + 'workday.csv'
      os.remove(my_file)
      new_index = range(len(zhibiao1))
      zhibiao1.index = new_index   
      zhibiao1.to_csv(father_path + 'workday.csv', index=False)
  else:
      zhibiao = pd.read_csv(father_path + 'holiday.csv')
      zhibiao1 = zhibiao.append(zhibiao_oneday)
      my_file = father_path + 'holiday.csv'
      os.remove(my_file)
      new_index = range(len(zhibiao1))
      zhibiao1.index = new_index   
      zhibiao1.to_csv(father_path + 'holiday.csv', index=False)
      
  
  
    #date_input = 
  date_index = date_list_1(zhibiao)
  zhibiao.index = date_index 
  zhibiao['time'] = date_index   

  #zhibiao1.to_csv("F:/work/tianhe4location/test/wori.csv")    
  date_index_list = sorted(list(set(list(date_index))), reverse=True)
  date_len = len(date_index_list)/24  
  if date_len <= 30:
      zhibiao_need_index = date_index_list
  else:
      zhibiao_need_index = date_index_list[:30*24]
          
  zhibiao_need = zhibiao.ix[zhibiao_need_index]
  zhibiao_need = zhibiao_need.sort_index()
    
  net_name = set(list(zhibiao_need.net_num))
  prediction_cell = list()
  predicted_last = pd.DataFrame(columns=['UE', 'erab', 'flow', 'handover', 
                                         'rrc', 'net_num'])
  
  for cell in net_name: 
      #reader_numpy = numpy.array(zhibiao[zhibiao.net_num =='GK477'][['UE', 'erab',
                                 #'flow', 'handover', 'rrc']])
    
      y_zhunbei = zhibiao_need[zhibiao_need.net_num == cell][['time',
                      'UE', 'erab', 'flow', 'handover', 'rrc']]
      y = numpy.array(y_zhunbei[['UE', 'erab', 'flow', 'handover', 
                                 'rrc']])
      #y = np.transpose(y_t)
      x = np.array(range(len(y)))
      
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
      
      predicted_out = DataFrame(prediction)
      predicted_out.columns = ['UE', 'erab', 'flow', 'handover', 'rrc']      
      predicted_out['net_num'] = cell
      predicted_last = predicted_last.append(predicted_out)
      
  my_pred_file = predicted_path + 'predicted_last.csv'
  os.remove(my_pred_file)
  predicted_last.to_csv(predicted_path  + 'predicted_last.csv', index=False)


