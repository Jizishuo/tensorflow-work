# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:27:25 2018

@author: Administrator
"""
import numpy as np
import pandas as pd

zhibiao = pd.read_csv('f:/work/data_month3_new.csv')
cell_net = pd.read_csv('f:/work/cell_200net.csv')
cell_net_need = cell_net[['Description', 'CellName']]
cell_net_need.columns = ['net_num', '小区名称']
zhibiao_merge = pd.merge(zhibiao, cell_net_need, on='小区名称', how='outer')   
    #zhibiao_clear = np.where(zhibiao['MR-总流量（KB）']==0)
    #zero_row = list(zhibiao_clear[0])
    #zhibiao_net_clear = zhibiao.drop(zero_row)
zhibiao_dropNa = zhibiao_merge.dropna()
zhibiao_date = zhibiao_dropNa['开始时间']
time_list = list()
day_list = list()
for time_ind in zhibiao_date:
    time_list.append(time_ind[11:13])
    day_list.append(time_ind[8:10])
    #zhibiao_dropNa.loc[zhibiao_index, 'hour'] = time_list
zhibiao_dropNa['hour'] = time_list
zhibiao_dropNa['day'] = day_list


zhibiao_dropNa = zhibiao_dropNa.drop('eNodeB名称', axis=1)
zhibiao_dropNa.index = zhibiao_dropNa['开始时间']
zhibiao_dropNa = zhibiao_dropNa.drop('开始时间', axis=1)

zhibiao_grouped = zhibiao_dropNa.groupby(['net_num', 'day', 'hour'])
zhibiao_liuliang= zhibiao_grouped['MR-总流量'].sum()
zhibiao_UE = zhibiao_grouped['MR-RRC连接建立最大用户数'].sum()

zhibiao1 = zhibiao_liuliang[:100000]
zhibiao2 = zhibiao_UE[:100000]

zhibiao1.to_csv('F:/work/test/zhibiao_liuliang.csv')
zhibiao2.to_csv('F:/work/test/zhibiao_UE.csv')
#liuliang_UE = zhibiao_liuliang/zhibiao_UE
#liuliang_UE.to_csv('F:/work/test/liuliang_UE.csv')




#zhibiao_new = pd.pivot_table(zhibiao_dropNa, index=['CELLID', 'day', 'hour'])['MR-总流量（KB）'].unstack()
#zhibiao_new = zhibiao_new/1024**2
#zhibiao_new.to_csv('F:/work/zhibiao_new.csv')
#zhibiao_grouped = zhibiao_dropNa.groupby(['net_num', 'day', 'hour'])['MR-总流量（KB）'].sum().unstack()
#zhibiao_grouped = zhibiao_grouped/1024**2
#zhibiao_grouped.to_csv('F:/work/zhibiao_grouped_new.csv')

#zhibiao_grouped_skew = zhibiao_dropNa.groupby(['net_num', 'day', 'hour'])['MR-总流量（KB）'].skew().unstack()
#zhibiao_grouped_skew.to_csv('F:/work/zhibiao_grouped_skew.csv')