# -*- coding: utf-8 -*-
"""
Created on Mon May  7 15:08:13 2018

@author: User
"""

import numpy as np
import pandas as pd
#data1 = pd.read_csv('F:/work/newdata.csv')
#data1_cell = list(set(list(data1['小区名称'])))

data = pd.read_csv('F:/work/test/month4_data_last.csv')
cell_net = pd.read_csv('F:/work/cell_200net.csv')
cell_net_need = cell_net[['Description', 'CellName']]
cell_net_need.columns = ['net_num', 'cellname']

zhibiao_merge = pd.merge(data, cell_net_need, on='cellname', how='outer')  
zhibiao_dropNa = zhibiao_merge.dropna()
zhibiao_dropNa.columns = ['haha', 'time', 'cellname', 'UE', 'qiehuanlv', 'liuliang',
       'rrcfenzhi', 'rrcfenmu', 'erabfenzi', 'erabfenmu', 'qiehuanfenzhi',
       'qiehuanfenmu', 'PRB_down', 'PRB_down_used', 'net_num']
zhibiao_dropNa.drop('haha', axis=1)

zhibiao_date = zhibiao_dropNa['time']
time_list = list()
day_list = list()
for time_ind in zhibiao_date:
    time_list.append(time_ind[11:13])
    day_list.append(time_ind[8:10])
    #zhibiao_dropNa.loc[zhibiao_index, 'hour'] = time_list
zhibiao_dropNa['hour'] = time_list
zhibiao_dropNa['day'] = day_list


zhibiao_dropNa = zhibiao_dropNa.drop('haha', axis=1)
zhibiao_dropNa.index = zhibiao_dropNa['time']
zhibiao_dropNa = zhibiao_dropNa.drop('time', axis=1)
zhibiao_dropNa = zhibiao_dropNa.drop('qiehuanlv', axis=1)


zhibiao_dropNa.index = zhibiao_dropNa['cellname']

#zhibiao_dropNa = zhibiao_dropNa.ix[data1_cell]




zhibiao_grouped = zhibiao_dropNa.groupby(['net_num', 'day', 'hour'])
#zhibiao_grouped = zhibiao_dropNa.groupby(['net_num', 'hour', 'day'])

zhibiao_liuliang = zhibiao_grouped['liuliang'].sum().unstack()
#zhibiao_liuliang = zhibiao_liuliang/1024**2

#zhibiao_liuliang.to_csv('F:/work/ririririrri.csv')


#zhibiao_rrc = zhibiao_grouped['rrcfenzhi'].sum().unstack
#zhibiao_erab = zhibiao_grouped['erabfenzi'].sum().unstack()
#zhibiao_qiehuan = zhibiao_grouped['qiehuanfenzhi'].sum().unstack()
#zhibiao_prb = zhibiao_grouped['PRB_down_used'].sum().unstack()
#zhibiao_ue = zhibiao_grouped['UE'].sum().unstack()


zhibiao1_liuliang= zhibiao_grouped['liuliang'].sum()
zhibiao1_rrc = zhibiao_grouped['rrcfenzhi'].sum()
zhibiao1_erab = zhibiao_grouped['erabfenzi'].sum()
zhibiao1_qiehuan = zhibiao_grouped['qiehuanfenzhi'].sum()
zhibiao1_prb = zhibiao_grouped['PRB_down_used'].sum()
zhibiao1_ue = zhibiao_grouped['UE'].sum()



#zhibiao_liuliang = zhibiao_liuliang/1024**2
#zhibiao_erab = zhibiao_erab/1024
#zhibiao_prb = zhibiao_prb/1024**2/10
#zhibiao_qiehuan = zhibiao_qiehuan/1024
#zhibiao_ue = zhibiao_ue/10
#zhibiao_rrc = zhibiao_rrc/1024

#def edge_compute(data, times):
#    data_median = data.median(axis=1)
#    data_std = data.std(axis=1)
#    data_edge = data_median - data_std*times
#    return data_edge

#judge_times = 2

#liuliang_edge = edge_compute(zhibiao_liuliang, judge_times)
#erab_edge = edge_compute(zhibiao_erab, judge_times)
#prb_edge = edge_compute(zhibiao_prb, judge_times)
#qiehuan_edge = edge_compute(zhibiao_qiehuan, judge_times)
#ue_edge = edge_compute(zhibiao_ue, judge_times)




zhibiao1_liuliang = zhibiao1_liuliang/1024**2
#zhibiao1_rrc = zhibiao1_rrc/1024
zhibiao1_erab = zhibiao1_erab/1024
zhibiao1_prb = zhibiao1_prb/1024**2/10
zhibiao1_qiehuan = zhibiao1_qiehuan/1024
zhibiao1_ue = zhibiao1_ue/10

result = pd.DataFrame({'flow':zhibiao1_liuliang, 'erab':zhibiao1_erab, 'UE':zhibiao1_ue, 
                       'qiehuan':zhibiao1_qiehuan, 'prb':zhibiao1_prb, 'rrc':zhibiao1_rrc})
    
result.to_csv('F:/work/test/test1/result.csv')


#zhibiao_liuliang.to_csv('F:/work/test/test1/zhibiao_liuliang.csv')
#zhibiao_rrc.to_csv('F:/work/test/test1/zhibiao_rrc.csv')
#zhibiao_erab.to_csv('F:/work/test/test1/zhibiao_erab.csv')
#zhibiao_qiehuan.to_csv('F:/work/test/test1/zhibiao_qiehuan.csv')
#zhibiao_prb.to_csv('F:/work/test/test1/zhibiao_prb.csv')
#zhibiao_ue.to_csv('F:/work/test/test1/zhibiao_ue.csv')

#zhibiao1_liuliang['HA480'].to_csv('F:/work/test/test2/zhibiao1_liuliang.csv')
#zhibiao_rrc.to_csv('F:/work/test/test1/zhibiao_rrc.csv')
#zhibiao1_erab['HA480'].to_csv('F:/work/test/test2/zhibiao1_erab.csv')
#zhibiao1_qiehuan['HA480'].to_csv('F:/work/test/test2/zhibiao1_qiehuan.csv')
#zhibiao1_prb['HA480'].to_csv('F:/work/test/test2/zhibiao1_prb.csv')
#zhibiao1_ue['HA480'].to_csv('F:/work/test/test2/zhibiao1_ue.csv')
