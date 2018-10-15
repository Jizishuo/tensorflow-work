# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:05:02 2018

@author: User
"""

#import numpy as np
import pandas as pd
import datetime
from pandas import DataFrame
import os
import sys


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

    
    
def data_full(da_input, date_lost_list_netnum):
    zhibiao_out = da_input
    date_list = list(set(list(zhibiao_out['time'])))
    lost_list = date_lost_list_netnum['cellname']
    lost_net_num = date_lost_list_netnum['net_num']
    dict_lost = dict(zip(lost_list, lost_net_num))
    zhibiao_out_last = DataFrame(columns=zhibiao_out.columns)
    #da_input['time'] = date_list_need
    for item in lost_list:
        item_date = list(zhibiao_out[zhibiao_out.cellname == item]['time'])
        item_date_lost = list(set(date_list).difference(set(item_date)))
        item_full = DataFrame(columns=zhibiao_out.columns)
        item_full['time'] = item_date_lost
        item_full['cellname'] = item
        item_full['net_num'] = dict_lost[item]
        item_full = item_full.fillna(0)
        zhibiao_out_last = zhibiao_out_last.append(item_full)
    zhibiao_out_haha = zhibiao_out.append(zhibiao_out_last)
    zhibiao_out_last_haha = date_caculate(zhibiao_out_haha)
    
    
    
    return zhibiao_out_last_haha
    
def wanzhengxing(da_input, cell_name):
    date_haha = da_input['time']
    dates_list = list(set(list(date_haha)))
    days = len(dates_list)/24
    date_numbs = 24*days
    cell_list = list(cell_name['cellname'])
    cell_list_input = list(da_input['cellname'])

    date_lost_list = list()
    date_lost_num = list()
    date_lost_out = DataFrame()
    
    for cell_item in cell_list:
        net_num_counter = cell_list_input.count(cell_item)
        if net_num_counter < date_numbs:
            date_lost_list.append(cell_item)
            date_lost_num.append(date_numbs-net_num_counter)
    
    date_lost_out['cellname'] = date_lost_list
    date_lost_out['lost_times'] = date_lost_num   
    return date_lost_out


def date_caculate(in_data):
    zhibiao_input = in_data
    zhibiao_date = zhibiao_input['time']
    time_list = list()
    day_list = list()
    month_list = list()
    year_list = list()
    for time_ind in zhibiao_date:
        time_list.append(time_ind[11:13])
        day_list.append(time_ind[8:10])
        month_list.append(time_ind[5:7])
        year_list.append(time_ind[0:4])
    #zhibiao_dropNa.loc[zhibiao_index, 'hour'] = time_list   
    zhibiao_input['hour'] = time_list
    zhibiao_input['day'] = day_list
    zhibiao_input['month'] = month_list
    zhibiao_input['year'] = year_list
    
    return zhibiao_input



def getExePath():
    sap = '/'
    if sys.argv[0].find(sap) == -1:
        sap = '\\'
    indx = sys.argv[0].rfind(sap)
    path = sys.argv[0][:indx] + sap
    return path


pwd = getExePath()

file_path = pwd  +'data' + '/'





#data1 = pd.read_csv('F:/work/tianhe4location/basic_data/orig_data/tianhe4location.csv')
#zhibiao_dropNa = data1[['开始时间', '小区名称', 'MR-RRC连接建立最大用户数',
                        #'YY-RRC连接建立成功率分母', 'YY-E-RAB连接建立成功率分母',
                        #'YY-切换成功率分母', 'MR-总流量（KB）', '所属网格']]


data1 = pd.read_csv(file_path +'today.csv')
data1_cell = list(set(list(data1['小区名称'])))

zhibiao_dropNa = data1[['开始时间', '小区名称', 'MR-RRC连接建立最大用户数_1437649632929',
                        'YY-RRC连接建立成功率分母', 'YY-E-RAB连接建立成功率分母',
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

zhibiao_dropNa_ca = date_caculate(zhibiao_dropNa)

full_cellname = pd.read_csv(file_path + 'cellname_4loc.csv')

cell_net = pd.read_csv(file_path + 'cell_200net.csv')
cell_net_need = cell_net[['Description', 'CellName']]
cell_net_need.columns = ['net_num', 'cellname']

date_lost_list = wanzhengxing(zhibiao_dropNa_ca, full_cellname)

date_lost_list_netnum = pd.merge(date_lost_list, cell_net_need, on='cellname', how='inner')


zhibiao_full_last = data_full(zhibiao_dropNa_ca, date_lost_list_netnum)


#wori = wanzhengxing(zhibiao_full_last, full_cellname)




#zhibiao_dropNa_full.to_csv('F:/work/tianhe4location/test/newdata.csv')
#zhibiao_dropNa = zhibiao_dropNa.drop('haha', axis=1)
#zhibiao_dropNa.index = zhibiao_dropNa['time']
#zhibiao_dropNa = zhibiao_dropNa.drop('time', axis=1)
#zhibiao_dropNa = zhibiao_dropNa.drop('qiehuanlv', axis=1)


#zhibiao_dropNa.index = zhibiao_dropNa['cellname']

#zhibiao_dropNa = zhibiao_dropNa.ix[data1_cell]

zhibiao_grouped = zhibiao_full_last.groupby(['net_num', 'year', 'month', 'day', 'hour'])
#zhibiao_grouped = zhibiao_dropNa.groupby(['net_num', 'hour', 'day'])

zhibiao_flow = zhibiao_grouped['flow'].sum()
zhibiao_rrc = zhibiao_grouped['rrc'].sum()
zhibiao_erab = zhibiao_grouped['erab'].sum()
zhibiao_handover = zhibiao_grouped['handover'].sum()
zhibiao_ue = zhibiao_grouped['ue'].sum()

zhibiao_flow = zhibiao_flow/1024**2
zhibiao_rrc = zhibiao_rrc/1024
zhibiao_erab = zhibiao_erab/1024
zhibiao_handover = zhibiao_handover/1024
zhibiao_ue = zhibiao_ue/10

result = pd.DataFrame({'flow':zhibiao_flow, 'erab':zhibiao_erab, 'UE':zhibiao_ue, 
                       'handover':zhibiao_handover, 'rrc':zhibiao_rrc})
    
    

#result.to_csv('F:/work/tianhe4location/result.csv')    
result.to_csv(file_path + 'result_today.csv')



#zhibiao_liuliang = zhibiao_liuliang/1024**2

#zhibiao_liuliang.to_csv('F:/work/tianhe4location/zhibiao_liuliang.csv')