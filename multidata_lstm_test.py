# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:41:03 2018

@author: User
"""

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pandas import DataFrame


from os import path
 
import numpy 
import tensorflow as tf
import pandas as pd
from pandas import DataFrame, Series

from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import model as ts_model
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

import datetime
import os

#import json  
#import urllib

#def holiday(date_list):
    #server_url = "http://www.easybots.cn/api/holiday.php?d="  
    #vop_response = urllib.request.urlopen(server_url + date_list)  
    #vop_data= json.loads(vop_response.read())
    #is_holiday = vop_data[date_list]
    #return is_holiday

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


    
def isworkday(data):
    date_thisday = data[0]
    if date_thisday in workday_2018:
        return "1"
    else:
        return "0"
    
    
def data_full(da_input, date_lost_list)   :
    date_list = list(date_list_1(zhibiao))
    zhibiao['time'] = date_list
    for item in date_lost_list:
        item_date = list(zhibiao[zhibiao.net_num == 'HH466']['time'])
        item_date_lost = list(set(date_list).difference(set(item_date)))
        item_full = DataFrame(columns=zhibiao.columns)
        item_full['time'] = item_date_lost
        item_full['net_num'] = 'HH466'
        item_full = item_full.dropna(0)
        
        
        
    
    
    
def wanzhengxing(da_input):
    date_haha = date_list(da_input)
    dates_list = list(set(list(date_haha)))
    days = len(dates_list)
    date_numbs = 24*days
    net_num_list = list(set(list(da_input['net_num'])))
    net_num_list_input = list(da_input['net_num'])
    
    date_lost_list = list()
    
    for net_item in net_num_list:
        net_num_counter = net_num_list_input.count(net_item)
        if net_num_counter < date_numbs:
            date_lost_list.append(net_item)
    
    for item in date_lost_list:
        
        
    return date_lost_list   




       
class _LSTMModel(ts_model.SequentialTimeSeriesModel):
  """A time series model-building example using an RNNCell."""

  def __init__(self, num_units, num_features, dtype=tf.float32):
    """Initialize/configure the model object.
    Note that we do not start graph building here. Rather, this object is a
    configurable factory for TensorFlow graphs which are run by an Estimator.
    Args:
      num_units: The number of units in the model's LSTMCell.
      num_features: The dimensionality of the time series (features per
        timestep).
      dtype: The floating point data type to use.
    """
    super(_LSTMModel, self).__init__(
        # Pre-register the metrics we'll be outputting (just a mean here).
        train_output_names=["mean"],
        predict_output_names=["mean"],
        num_features=num_features,
        dtype=dtype)
    self._num_units = num_units
    # Filled in by initialize_graph()
    self._lstm_cell = None
    self._lstm_cell_run = None
    self._predict_from_lstm_output = None

  def initialize_graph(self, input_statistics):
    """Save templates for components, which can then be used repeatedly.
    This method is called every time a new graph is created. It's safe to start
    adding ops to the current default graph here, but the graph should be
    constructed from scratch.
    Args:
      input_statistics: A math_utils.InputStatistics object.
    """
    super(_LSTMModel, self).initialize_graph(input_statistics=input_statistics)
    self._lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self._num_units)
    # Create templates so we don't have to worry about variable reuse.
    self._lstm_cell_run = tf.make_template(
        name_="lstm_cell",
        func_=self._lstm_cell,
        create_scope_now_=True)
    # Transforms LSTM output into mean predictions.
    self._predict_from_lstm_output = tf.make_template(
        name_="predict_from_lstm_output",
        func_=lambda inputs: tf.layers.dense(inputs=inputs, units=self.num_features),
        create_scope_now_=True)

  def get_start_state(self):
    """Return initial state for the time series model."""
    return (
        # Keeps track of the time associated with this state for error checking.
        tf.zeros([], dtype=tf.int64),
        # The previous observation or prediction.
        tf.zeros([self.num_features], dtype=self.dtype),
        # The state of the RNNCell (batch dimension removed since this parent
        # class will broadcast).
        [tf.squeeze(state_element, axis=0)
         for state_element
         in self._lstm_cell.zero_state(batch_size=1, dtype=self.dtype)])

  def _transform(self, data):
    """Normalize data based on input statistics to encourage stable training."""
    mean, variance = self._input_statistics.overall_feature_moments
    return (data - mean) / variance

  def _de_transform(self, data):
    """Transform data back to the input scale."""
    mean, variance = self._input_statistics.overall_feature_moments
    return data * variance + mean

  def _filtering_step(self, current_times, current_values, state, predictions):
    """Update model state based on observations.
    Note that we don't do much here aside from computing a loss. In this case
    it's easier to update the RNN state in _prediction_step, since that covers
    running the RNN both on observations (from this method) and our own
    predictions. This distinction can be important for probabilistic models,
    where repeatedly predicting without filtering should lead to low-confidence
    predictions.
    Args:
      current_times: A [batch size] integer Tensor.
      current_values: A [batch size, self.num_features] floating point Tensor
        with new observations.
      state: The model's state tuple.
      predictions: The output of the previous `_prediction_step`.
    Returns:
      A tuple of new state and a predictions dictionary updated to include a
      loss (note that we could also return other measures of goodness of fit,
      although only "loss" will be optimized).
    """
    state_from_time, prediction, lstm_state = state
    with tf.control_dependencies(
            [tf.assert_equal(current_times, state_from_time)]):
      transformed_values = self._transform(current_values)
      # Use mean squared error across features for the loss.
      predictions["loss"] = tf.reduce_mean(
          (prediction - transformed_values) ** 2, axis=-1)
      # Keep track of the new observation in model state. It won't be run
      # through the LSTM until the next _imputation_step.
      new_state_tuple = (current_times, transformed_values, lstm_state)
    return (new_state_tuple, predictions)

  def _prediction_step(self, current_times, state):
    """Advance the RNN state using a previous observation or prediction."""
    _, previous_observation_or_prediction, lstm_state = state
    lstm_output, new_lstm_state = self._lstm_cell_run(
        inputs=previous_observation_or_prediction, state=lstm_state)
    next_prediction = self._predict_from_lstm_output(lstm_output)
    new_state_tuple = (current_times, next_prediction, new_lstm_state)
    return new_state_tuple, {"mean": self._de_transform(next_prediction)}

  def _imputation_step(self, current_times, state):
    """Advance model state across a gap."""
    # Does not do anything special if we're jumping across a gap. More advanced
    # models, especially probabilistic ones, would want a special case that
    # depends on the gap size.
    return state

  def _exogenous_input_step(
          self, current_times, current_exogenous_regressors, state):
    """Update model state based on exogenous regressors."""
    raise NotImplementedError(
        "Exogenous inputs are not implemented for this example.")


def ratio_mark(data_input, standard):
    div_mod = divmod(data_input, standard)
    ratio_mark_out = abs(div_mod[0]-1)*div_mod[1]/standard
    ratio_mark_out = ratio_mark_out.replace(0, 1)
    return ratio_mark_out
    
    


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
                
    flow_ratio = data_input_ratio['flow']
    flow_ratio.index = range(len(flow_ratio))
    flow_data_pred = data_input['flow']    
    flow_data_pred.index = range(len(flow_data_pred))  
    last_flow = lastday_input['flow']
    
    
    flow_mark = ratio_mark(flow_data_pred, flow_standard)# 超过3G则不考虑比例转换
    flow_ratio_out = flow_ratio*flow_mark
    
    flow_isok = list(map(lambda y:0  if y>alert_standard else 1, flow_ratio_out))
    
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
        plt.savefig('F:/work/tianhe4location/output/picture/' + net_item +'.jpg')    
    
    return ratio_out
    


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  #计算当天指标是否有异常
  predicted_zhibiao = pd.read_csv('F:/work/tianhe4location/output/predicted_last.csv')
  predicted_zhibiao.columns = ['hour', 'UE', 'erab', 'flow', 'handover', 'rrc', 'net_num']
  zhibiao_lastday = pd.read_csv("F:/work/tianhe4location/20180521.csv")
  
  zhibiao_last_day = zhibiao_lastday[['hour', 'UE' , 'erab', 'flow', 'handover', 'rrc', 'net_num']] 
  net_num_list = list(set(list(predicted_zhibiao['net_num'])))
  
  bijiao_result = DataFrame()
  
  for net_item in net_num_list:
      pred_zhibiao = predicted_zhibiao[predicted_zhibiao.net_num == net_item][
              ['hour', 'UE' , 'erab', 'flow', 'handover', 'rrc']]
      last_zhibiao = zhibiao_last_day[zhibiao_last_day.net_num == net_item][[
              'hour', 'UE' , 'erab', 'flow', 'handover', 'rrc']]
      #bijiao = DataFrame(np.array(pred_zhibiao) - np.array(last_zhibiao), 
                         #columns=['hour', 'UE' , 'erab', 'flow', 'handover', 'rrc'])
      bijiao = np.array(pred_zhibiao) - np.array(last_zhibiao)                                         
      bijiao_ratio = np.array(bijiao)/np.array(pred_zhibiao)
      
      bijiao_da = DataFrame(bijiao, columns=['hour', 'UE' , 'erab', 
                         'flow', 'handover', 'rrc'])
      bijiao_ratio_da = DataFrame(bijiao_ratio, columns=['hour', 'UE' , 'erab', 
                   'flow', 'handover', 'rrc'])
           
      bijiao_notok = not_ok(pred_zhibiao, last_zhibiao, bijiao_ratio_da, net_item, 5, 0.5) # 3表示比例转换的流量基准， 0.5表示告警门限 
      bijiao_notok['net_num'] = net_item
      bijiao_result = bijiao_result.append(bijiao_notok) 
      
  #bijiao_result.to_csv("F:/work/tianhe4location/output/isok.csv")   
  
  
  #计算是否为公休日
  workday_calender = pd.read_csv("F:/work/tianhe4location/basic_data/2018workday.csv")
  workday_2018 = str2time(list(workday_calender['date']))   
  zhibiao_oneday = pd.read_csv("F:/work/tianhe4location/20180521.csv")
  data = date_list(zhibiao_oneday)
  is_workday = isworkday(data)
  
  if is_workday:
      zhibiao = pd.read_csv("F:/work/tianhe4location/workday.csv")
      zhibiao1 = zhibiao.append(zhibiao_oneday)
      my_file = "F:/work/tianhe4location/workday.csv"
      os.remove(my_file)
      new_index = range(len(zhibiao1))
      zhibiao1.index = new_index   
      zhibiao1.to_csv("F:/work/tianhe4location/workday.csv")
  else:
      zhibiao = pd.read_csv("F:/work/tianhe4location/holiday.csv")
      zhibiao1 = zhibiao.append(zhibiao_oneday)
      my_file = "F:/work/tianhe4location/holiday.csv"
      os.remove(my_file)
      new_index = range(len(zhibiao1))
      zhibiao1.index = new_index   
      zhibiao1.to_csv("F:/work/tianhe4location/holiday.csv")
      
  
  
    #date_input = 
  date_index = date_list(zhibiao)
  zhibiao.index = date_index    

  #zhibiao1.to_csv("F:/work/tianhe4location/test/wori.csv")    
  date_index_list = sorted(list(set(list(date_index))), reverse=True)
  date_len = len(date_index_list)   
  if date_len <= 30:
      zhibiao_need_index = date_index_list
  else:
      zhibiao_need_index = date_index_list[:30]
          
  zhibiao_need = zhibiao.ix[zhibiao_need_index]
  zhibiao_need = zhibiao_need.sort_index()
    
  net_name = set(list(zhibiao_need.net_num))
  prediction_cell = list()
  predicted_last = pd.DataFrame(columns=['UE', 'erab', 'flow', 'handover', 'rrc', 'net_num'])
  
  for cell in net_name: 
      #reader_numpy = numpy.array(zhibiao[zhibiao.net_num =='GK477'][['UE', 'erab',
                                 #'flow', 'handover', 'rrc']])
    
      y = numpy.array(zhibiao_need[zhibiao_need.net_num ==cell][['UE', 'erab',
                                 'flow', 'handover', 'rrc']])
      x = np.array(range(len(y)))
      data = {tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
              tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,}
      reader = NumpyReader(data)
    
    
    
      #reader = tf.contrib.timeseries.CSVReader(
              #csv_file_name,
              #column_names=((tf.contrib.timeseries.TrainEvalFeatures.TIMES,)
                    #+ (tf.contrib.timeseries.TrainEvalFeatures.VALUES,) * 1))
      train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
              reader, batch_size=20, window_size=168)

      estimator = ts_estimators.TimeSeriesRegressor(
              model=_LSTMModel(num_features=5, num_units=96),
              optimizer=tf.train.AdamOptimizer(0.001))

      estimator.train(input_fn=train_input_fn, steps=200)
      evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
      evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
  # Predict starting after the evaluation
      (predictions,) = tuple(estimator.predict(
              input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
                      evaluation, steps=24)))

      observed_times = evaluation["times"][0]
      observed = evaluation["observed"][0, :, :]
      evaluated_times = evaluation["times"][0]
      evaluated = evaluation["mean"][0]
      predicted_times = predictions['times']
      predicted = predictions["mean"]
      
      predicted_out = DataFrame(predicted)
      predicted_out.columns = ['UE', 'erab', 'flow', 'handover', 'rrc']
      predicted_out['net_num'] = cell
      predicted_last = predicted_last.append(predicted_out)
      
  my_pred_file = 'F:/work/tianhe4location/output/predicted_last.csv'
  os.remove(my_pred_file)
  predicted_last.to_csv('F:/work/tianhe4location/output/predicted_last.csv')
      #plt.figure(figsize=(30, 8))
      #plt.axvline(99, linestyle="dotted", linewidth=4, color='r')
      #observed_lines = plt.plot(observed_times, observed, label="observation", color="k")
      #evaluated_lines = plt.plot(evaluated_times, evaluated, label="evaluation", color="g")
      #predicted_lines = plt.plot(predicted_times, predicted, label="prediction", color="r")
      #plt.legend(handles=[observed_lines[0], evaluated_lines[0], predicted_lines[0]],
                 #loc="upper left")
  #plt.savefig('F:/work/TFTS/data/output/predict_result_2.jpg')
