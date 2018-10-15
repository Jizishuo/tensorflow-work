# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:27:25 2018

@author: Administrator
"""
import numpy as np
import pandas as pd

chunks_train = pd.read_csv('D:/data/M3_cell_utf8.csv', chunksize = 10000000)
cell_net = pd.read_csv('D:/data/cell_net_500m.csv')
zhibiao_drop_clear = list()
raw_index = 1
for raw in chunks_train:
    #清除空值和流量为0的项
    zhibiao = pd.merge(raw, cell_net, on='CELLID', how='outer')   
    #zhibiao_clear = np.where(zhibiao['MR-总流量（KB）']==0)
    #zero_row = list(zhibiao_clear[0])
    #zhibiao_net_clear = zhibiao.drop(zero_row)
    #zhibiao_dropNa = zhibiao_net_clear.dropna()
    zhibiao_dropNa = zhibiao.dropna()
    
    #截取小时数
    #zhibiao_index = zhibiao_dropNa.index
    zhibiao_date = zhibiao_dropNa['开始时间']
    time_list = list()
    day_list = list()
    for time_ind in zhibiao_date:
        time_list.append(time_ind[11:13])
        day_list.append(time_ind[8:10])
    #zhibiao_dropNa.loc[zhibiao_index, 'hour'] = time_list
    zhibiao_dropNa['hour'] = time_list
    zhibiao_dropNa['day'] = day_list
    
    #分别读入磁盘，对于第一个chunk，用方式‘W’，后面的chunks用‘A’方式
    if raw_index == 1:
        zhibiao_dropNa.to_csv('D:/data/test/test3.csv', index = False, mode = 'w') 
        raw_index = raw_index + 1
    else:
        zhibiao_dropNa.to_csv('D:/data/test/test3.csv', index = False, header = False, mode = 'a') 
        raw_index = raw_index + 1

#Try using .loc[row_indexer,col_indexer] = value instead
    
               
chunks_train = pd.read_csv('D:/data/TEST03/test03new.csv', chunksize = 1000000)
raw_index = 1
for raw in chunks_train:
    #去掉原来时间那一列，网格号取整数， 流量取G保留2位小数
    raw.drop('开始时间',axis=1, inplace=True)
    #raw_new['cellid'] = raw_new['cellid'].astype(np.int16)
    raw['MR-总流量（KB）'] = round(raw['MR-总流量（KB）']/1024**2, 3)
    
    #分别读入磁盘，对于第一个chunk，用方式‘W’，后面的chunks用‘A’方式
    if raw_index == 1:
        raw.to_csv('D:/data/test/test03_last.csv', index = False, mode = 'w') 
        raw_index = raw_index + 1
    else:
        raw.to_csv('D:/data/test/test03_last.csv', index = False, header = False, mode = 'a') 
        raw_index = raw_index + 1
