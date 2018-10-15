# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:34:20 2018

@author: User
"""
import pandas as pd  
import numpy as np  
import tensorflow as tf  
from sklearn.metrics import mean_absolute_error,mean_squared_error  
from sklearn.preprocessing import MinMaxScaler  
import matplotlib.pyplot as plt  
import warnings   

data = pd.read_csv('F:/work/test/test1.csv')
data_train = data['流量']

#定义常量  
rnn_unit=10       #hidden layer units  
input_size=4        
output_size=1  
lr=0.0006         #学习率  
tf.reset_default_graph()  
#输入层、输出层权重、偏置  
weights={  
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),  
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))  
         }  
biases={  
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),  
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))  
        }  



#分割数据集，将数据分为训练集和验证集（最后90天做验证，其他做训练）
def get_data(batch_size=60,time_step=20,train_begin=0,train_end=743):  
    batch_index=[]  
          
    scaler_for_x=MinMaxScaler(feature_range=(0,1))  #按列做minmax缩放  
    scaler_for_y=MinMaxScaler(feature_range=(0,1))  
    scaled_x_data=scaler_for_x.fit_transform(data[:,:-1])  
    scaled_y_data=scaler_for_y.fit_transform(data[:,-1])  
      
    label_train = scaled_y_data[train_begin:train_end]  
    label_test = scaled_y_data[train_end:]  
    normalized_train_data = scaled_x_data[train_begin:train_end]  
    normalized_test_data = scaled_x_data[train_end:]  
      
    train_x,train_y=[],[]   #训练集x和y初定义  
    for i in range(len(normalized_train_data)-time_step):  
        if i % batch_size==0:  
            batch_index.append(i)  
        x=normalized_train_data[i:i+time_step,:4]  
        y=label_train[i:i+time_step,np.newaxis]  
        train_x.append(x.tolist())  
        train_y.append(y.tolist())  
    batch_index.append((len(normalized_train_data)-time_step))  
      
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample   
    test_x,test_y=[],[]    
    for i in range(size-1):  
        x=normalized_test_data[i*time_step:(i+1)*time_step,:4]  
        y=label_test[i*time_step:(i+1)*time_step]  
        test_x.append(x.tolist())  
        test_y.extend(y)  
    test_x.append((normalized_test_data[(i+1)*time_step:,:4]).tolist())  
    test_y.extend((label_test[(i+1)*time_step:]).tolist())      
      
    return batch_index,train_x,train_y,test_x,test_y,scaler_for_y 

#定义lstm网络
#——————————————————定义神经网络变量——————————————————  
def lstm(X):    
    batch_size=tf.shape(X)[0]  
    time_step=tf.shape(X)[1]  
    w_in=weights['in']  
    b_in=biases['in']    
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入  
    input_rnn=tf.matmul(input,w_in)+b_in  
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入  
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)  
    #cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)  
    init_state=cell.zero_state(batch_size,dtype=tf.float32)  
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果  
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入  
    w_out=weights['out']  
    b_out=biases['out']  
    pred=tf.matmul(output,w_out)+b_out  
    return pred,final_states  


#模型训练与预测
#——————————————————训练模型——————————————————  
def train_lstm(batch_size=80,time_step=15,train_begin=0,train_end=487):  
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])  
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])  
    batch_index,train_x,train_y,test_x,test_y,scaler_for_y = get_data(batch_size,time_step,train_begin,train_end)  
    pred,_=lstm(X)  
    #损失函数  
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))  
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)    
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())  
        #重复训练5000次  
        iter_time = 5000  
        for i in range(iter_time):  
            for step in range(len(batch_index)-1):  
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})  
            if i % 100 == 0:      
                print('iter:',i,'loss:',loss_)  
        ####predict####  
        test_predict=[]  
        for step in range(len(test_x)):  
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})     
            predict=prob.reshape((-1))  
            test_predict.extend(predict)  
              
        test_predict = scaler_for_y.inverse_transform(test_predict)  
        test_y = scaler_for_y.inverse_transform(test_y)  
        rmse=np.sqrt(mean_squared_error(test_predict,test_y))  
        mae = mean_absolute_error(y_pred=test_predict,y_true=test_y)  
        print ('mae:',mae,'   rmse:',rmse)  
    return test_predict  


#调用train_lstm()函数，完成模型训练与预测的过程，并统计验证误差（mae和rmse）:
test_predict = train_lstm(batch_size=80,time_step=15,train_begin=0,train_end=487) 



