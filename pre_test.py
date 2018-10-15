# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:38:16 2018

@author: User
"""

#from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader
import pymysql

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from pandas import DataFrame

#from os import path
import numpy 
from pandas import DataFrame
import matplotlib
import datetime
import os
import sys
#import data_done as done
from sqlalchemy import create_engine

from multiprocessing import Process, Pool
import time

class predict(object):
    
    def __init__(self):
        self.__pwd = self.__getExePath()
        self.__predicted_path = self.__pwd +'output' + os.path.sep
        self.__predicted_picture_ok_path = self.__pwd +'output' + os.path.sep + 'picture' + os.path.sep + 'ok' + os.path.sep
        self.__predicted_picture_notok_path = self.__pwd +'output' + os.path.sep + 'picture' + os.path.sep + 'notok' + os.path.sep
        self.__data_basic_path = self.__pwd +'basic_data' + os.path.sep 
        self.__my_pred_file_workday = self.__predicted_path + 'predicted_last_workday.csv'
        self.__my_pred_file_holiday = self.__predicted_path + 'predicted_last_holiday.csv'
        self.zhibiao_need = DataFrame()
        

    
    def __getExePath(self):
        sap = '/'
        if sys.argv[0].find(sap) == -1:
            sap = '\\'
        indx = sys.argv[0].rfind(sap)
        path = sys.argv[0][:indx] + sap
        return path

    

    def __str2time_1(self, list_in):
        date_list_new = list()
        for time_id in list_in:
            date_new = datetime.datetime.strptime(time_id,'%Y/%m/%d/%H')
            date_new = date_new.strftime('%Y-%m-%d %H:%M:%S.000')
            date_list_new.append(date_new)
        return date_list_new

    def __date_list_1(self, zhibiao):
        d_year = zhibiao['year']
        d_month = zhibiao['month']
        d_day = zhibiao['day']
        d_hour = zhibiao['hour']
        date_len = len(d_year)
        date_haha = range(date_len)
        date_list1 = list(map(lambda i :str(d_year[i])+ '/' + str(d_month[i]) + '/'
                              + str(d_day[i])+ '/' + str(d_hour[i]) , date_haha))
        date_list_out = self.__str2time_1(date_list1)
        return date_list_out

    def __str2time(self, list_in):
        date_list_new = list()
        for time_id in list_in:
            date_new = datetime.datetime.strptime(time_id,'%Y/%m/%d')
            date_new = date_new.strftime('%Y-%m-%d %H:%M:%S.000')
            date_list_new.append(date_new)
        return date_list_new

    def __date_list(self, zhibiao):
        d_year = zhibiao['year']
        d_month = zhibiao['month']
        d_day = zhibiao['day']
        date_len = len(d_year)
        date_haha = range(date_len)
        date_list1 = list(map(lambda i :str(d_year[i])+ '/' + str(d_month[i]) + '/'
                              + str(d_day[i]) , date_haha))
        date_list_out = self.__str2time(date_list1)
        return date_list_out
  
    def __isworkday(self, data):
        date_thisday = data[0]
        if date_thisday in date_thisday:
            return True
        else:
           return False
        
    def __ratio_mark(self, data_input, standard):
        data_input_list = DataFrame(data_input, columns=['flow'])
        mark_1 = abs(data_input_list.where(data_input_list['flow']<standard)/standard)
        mark_out = mark_1.fillna(1)
        return mark_out
    

    def __del_file(self, path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                self.__del_file(c_path)
            else:
                os.remove(c_path)

    def __not_ok(self, data_input, lastday_input, data_input_ratio, net_item, flow_standard, alert_standard):      
        ratio_need = data_input_ratio[['UE', 'erab', 'handover', 'rrc']]
        ratio_need_columns = ratio_need.columns
        ratio_out = DataFrame()
        for columns_need in ratio_need_columns:
            ratio_item = ratio_need[columns_need]
            ratio_isok = list(map(lambda y:0  if y>alert_standard else 1, ratio_item))
            ratio_out[columns_need] = ratio_isok
                
        flow_ratio = DataFrame(data_input_ratio['flow'],columns=['flow'])
        flow_ratio.index = range(len(flow_ratio))
        flow_data_pred = data_input['flow']    
        flow_data_pred.index = range(len(flow_data_pred))  
        last_flow = lastday_input['flow']    
        flow_mark = self.__ratio_mark(flow_data_pred, flow_standard)# 超过3G则不考虑比例转换
        flow_ratio_out = DataFrame(np.array(flow_ratio)*np.array(flow_mark), columns=['flow'])
        flow_ratio_out_series = flow_ratio_out['flow']    
        flow_isok = list(map(lambda y:0  if y>alert_standard else 1, flow_ratio_out_series))
        ratio_out['flow'] = flow_isok
    
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
            plt.savefig(self.__predicted_picture_notok_path + net_item +'.png')    
        else:        
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            ax.plot(range(24), flow_data_pred, label="flow_pred", color="g")
            ax.plot(range(24), last_flow, label="flow_lastday", color="r")
            ax.set_ylabel('GB')
            ax.set_xlabel('Hour')
            plt.legend(loc="upper left")
            plt.title('net_num: ' + net_item + '  flow lastday')
            plt.savefig(self.__predicted_picture_ok_path + net_item +'.png')      
        return ratio_out
    
    def tensor_flow_go(self, net_list):
        predicted_last = pd.DataFrame(columns=['UE', 'erab', 'flow', 'handover', 
                                               'rrc', 'net_num'])
        for cell in net_list: 
            y_zhunbei = self.zhibiao_need[self.zhibiao_need.net_num == cell][['time',
                                    'UE', 'erab', 'flow', 'handover', 'rrc']]
            y = numpy.array(y_zhunbei[['UE', 'erab', 'flow', 'handover', 
                                 'rrc']])
            x = np.array(range(len(y)))
      
            data = {tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
                    tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,}
            reader = NumpyReader(data)
      
            train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=10, 
                                                                       window_size=48)
            ar = tf.contrib.timeseries.ARRegressor(
                    periodicities=48, input_window_size=24, output_window_size=24,
                    num_features=5, loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)
    
            ar.train(input_fn=train_input_fn, steps=700)
    
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


    def predicter_tenorflow(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        conn = pymysql.connect("localhost", "root", "1234", "python_test", charset='utf8' )
        cursor = conn.cursor()
        cursor.execute("select * from python_test.2018workday")
        canlender_2018 = cursor.fetchall()
        workday_calender = DataFrame(list(canlender_2018), columns=['date'])
        cursor.close
        conn.close  
        workday_2018 = self.__str2time(list(workday_calender['date']))   
        #zhibiao_oneday = pd.read_csv(self.__pwd + 'today.csv')
        zhibiao_oneday = pd.read_csv('F:/work/pyqt/predicter/today.csv')
        
        data_1 = self.__date_list(zhibiao_oneday)
        is_workday = self.__isworkday(data_1) 
        print(is_workday)
               
        conn = pymysql.connect("localhost", "root", "1234", "python_test", charset='utf8' )
        cursor = conn.cursor()
        engine = create_engine('mysql+pymysql://root:1234@127.0.0.1:3306/python_test?charset=utf8')
        if is_workday:
            #从数据库读取workday
            cursor.execute("select * from python_test.workday")
            work_day = cursor.fetchall()
            zhibiao = DataFrame(list(work_day), columns=['net_num', 'year', 
                                'month', 'day','hour', 'UE', 'erab', 'flow',
                                'handover', 'rrc'])
            cursor.execute("drop table python_test.workday")
            zhibiao1 = zhibiao.append(zhibiao_oneday)
            zhibiao2 = zhibiao1.drop_duplicates(['net_num', 'year', 
                                'month', 'day','hour']) 
            zhibiao2.to_sql('workday',con=engine, schema='python_test', index=False,
                            index_label=False, if_exists='append', chunksize=1000)
            conn.commit()

        else:
            #从数据库读取holiday
            cursor.execute("select * from python_test.holiday")
            holi_day = cursor.fetchall()
            zhibiao = DataFrame(list(holi_day), columns=['net_num', 'year', 
                                'month', 'day','hour', 'UE', 'erab', 'flow',
                                'handover', 'rrc'])
            cursor.execute("drop table python_test.holiday")
            zhibiao1 = zhibiao.append(zhibiao_oneday)
            zhibiao2 = zhibiao1.drop_duplicates(['net_num', 'year', 
                                'month', 'day','hour']) 
            zhibiao2.to_sql('holiday',con=engine, schema='python_test', index=False,
                            index_label=False, if_exists='append', chunksize=1000)
            conn.commit()
        cursor.close()
        conn.close()
    
    
        date_index = self.__date_list_1(zhibiao)
        zhibiao.index = date_index 
        zhibiao['time'] = date_index   

        #zhibiao1.to_csv("F:/work/tianhe4location/test/wori.csv")   
  
        date_index_list = sorted(list(set(list(date_index))), reverse=True)
        date_len = len(date_index_list)/24  
        if date_len <= 30:
            zhibiao_need_index = date_index_list
        else:
            zhibiao_need_index = date_index_list[:60*24]
          
        self.zhibiao_need = zhibiao.ix[zhibiao_need_index]
        self.zhibiao_need = self.zhibiao_need.sort_index()
    
        net_name = set(list(self.zhibiao_need.net_num))
        net_name_all = list(net_name)
        #print(net_name_all)
        net_num_perg = round(len(net_name)/8)
        
        net_list_index = list()
        for i in range(8):
            if i == 7:
                net_list_i = net_name_all[i*net_num_perg:len(net_name)-1]
            else:
                net_list_i = net_name_all[i*net_num_perg:(i+1)*net_num_perg]
            net_list_index.append(net_list_i)
        
        predicted_all_last = pd.DataFrame(columns=['UE', 'erab', 'flow', 'handover', 
                                               'rrc', 'net_num'])
                
        xx =net_list_index[0]
        self.tensor_flow_go(xx)
       

      
    
        
        
        
            
        
        
        
        
        
        
        
        