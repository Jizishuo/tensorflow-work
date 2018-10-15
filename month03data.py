# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 15:19:12 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

zhibiao_hour = pd.read_csv('F:/work/test03_last_new.csv')
x = pd.pivot_table(zhibiao_hour, index=['net_num', 'DAY', 'hour'], aggfunc=DataFrame.kurt)
#grouped_data_cell = zhibiao_hour.groupby(['cellid', 'DAY', 'hour'])
#grouped_cell_zhibiao = grouped_data_cell['MR-总流量（KB）'].sum().unstack()


grouped_data = zhibiao_hour.groupby(['net_num', 'DAY', 'hour'])
grouped_zhibiao = grouped_data['MR-总流量（KB）'].skew().unstack()

weekend = (3, 4, 10, 11, 17, 18, 24, 25, 31)
month = list(np.arange(1, 32, 1))
workday =  list(set(month).difference(set(weekend))) 


                                  
chunks_train = pd.read_csv('D:/data/test03_last_new.csv', chunksize = 20000000)
zhibiao_drop_clear = list()
raw_index = 1
for raw in chunks_train:
    #清除空值和流量为0的项
    x = pd.pivot_table(raw, index=['net_num', 'DAY', 'hour'], aggfunc=DataFrame.kurt)
    y = x['MR-总流量（KB）']
    y = y.unstack()
    
    #分别读入磁盘，对于第一个chunk，用方式‘W’，后面的chunks用‘A’方式
    if raw_index == 1:
        y.to_csv('D:/data/test/kurt_data.csv', index = True, mode = 'w') 
        raw_index = raw_index + 1
    else:
        y.to_csv('D:/data/test/kurt_data.csv', index = True, header = False, mode = 'a') 
        raw_index = raw_index + 1