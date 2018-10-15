# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:05:02 2018

@author: User
"""

import numpy as np
import pandas as pd
import datetime
from pandas import DataFrame, Series
import os
import sys


def getExePath():
    sap = '/'
    if sys.argv[0].find(sap) == -1:
        sap = '\\'
    indx = sys.argv[0].rfind(sap)
    path = sys.argv[0][:indx] + sap
    return path

pwd = getExePath()

def str2time(list_in):
    date_list_new = list()
    for time_id in list_in:
        date_new = datetime.datetime.strptime(time_id,'%Y/%m/%d')
        date_new = date_new.strftime('%Y-%m-%d %H:%M:%S.000')
        date_list_new.append(date_new)
    return date_list_new

def isworkday(data):
    date_thisday = data[0]
    if date_thisday in workday_2018:
        return True
    else:
        return False


def standard_data(result, net_list, stand_times):
    
    data_standard = DataFrame(columns= ['net_num', 'hour', 'UE', 'erab', 'flow', 'handover', 'rrc'])

    hours = ['00', '01', '02', '03','04', '05', '06', '07', '08', '09', '10', 
             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
             '21', '22', '23']
    
    net_num = 'HH466'
    hour_num = '08'
    
    for net_num in net_list:
        for hour_num in hours:
            UE_median = result.loc[(net_num,slice(None),slice(None),slice(None),hour_num),'UE'].median()
            UE_std = result.loc[(net_num,slice(None),slice(None),slice(None),hour_num),'UE'].std()
            UE_standard = UE_median-stand_times*UE_std


            erab_median = result.loc[(net_num,slice(None),slice(None),slice(None),hour_num),'erab'].median()
            erab_std = result.loc[(net_num,slice(None),slice(None),slice(None),hour_num),'erab'].std()
            erab_standard = erab_median-stand_times*erab_std

            flow_median = result.loc[(net_num,slice(None),slice(None),slice(None),hour_num),'flow'].median()
            flow_std = result.loc[(net_num,slice(None),slice(None),slice(None),hour_num),'flow'].std()
            flow_standard = flow_median-stand_times*flow_std

            handover_median = result.loc[(net_num,slice(None),slice(None),slice(None),hour_num),'handover'].median()
            handover_std = result.loc[(net_num,slice(None),slice(None),slice(None),hour_num),'handover'].std()
            handover_standard = handover_median-stand_times*handover_std

            rrc_median = result.loc[(net_num,slice(None),slice(None),slice(None),hour_num),'rrc'].median()
            rrc_std = result.loc[(net_num,slice(None),slice(None),slice(None),hour_num),'rrc'].std()
            rrc_standard = rrc_median-stand_times*rrc_std

            data_list = list([net_num, hour_num, UE_standard, erab_standard, flow_standard, handover_standard, rrc_standard])
            data_list1 = DataFrame(data_list).T
            data_list1.columns = ['net_num', 'hour', 'UE', 'erab', 'flow', 'handover', 'rrc']            
            data_standard = data_standard.append(data_list1, ignore_index=True)
    
    return data_standard


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

#da_input = zhibiao_dropNa_ca
#cell_name = full_cellname

def wanzhengxing(da_input, cell_name):
    date_haha = da_input['time']
    dates_list = list(set(list(date_haha)))
    days = len(dates_list)/24
    date_numbs = 24*days
    cell_list = cell_name
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




#data1 = pd.read_csv('F:/work/tianhe4location/basic_data/orig_data/tianhe4location.csv')
#zhibiao_dropNa = data1[['开始时间', '小区名称', 'MR-RRC连接建立最大用户数',
                        #'YY-RRC连接建立成功率分母', 'YY-E-RAB连接建立成功率分母',
                        #'YY-切换成功率分母', 'MR-总流量（KB）', '所属网格']]

data1 = pd.read_csv('F:/VVIP/all_0730.csv')
data1_cell = list(set(list(data1['小区名称'])))

xx = data1[['开始时间', '小区名称', 'MR-RRC连接建立最大用户数',
                        'YY-RRC连接建立成功率分母', 'YY-E-RAB连接建立成功率分母',
                        'YY-切换成功率分母', 'MR-总流量（KB）', 'TCP二三步成功率',
                        'VOLTE呼叫占比', '大包速率Mbps', '小包时延ms', '所属网格']]
#zhibiao_dropNa = xx.dropna()
zhibiao_dropNa = xx.fillna(0)

zhibiao_dropNa.columns = ['time', 'cellname', 'ue', 'rrc', 'erab', 'handover', 
                          'flow', 'tcp', 'volte', 'big_rate', 'small_delay', 'net_num']



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
full_cellname = list(set(zhibiao_dropNa['cellname']))

#full_cellname1 = pd.read_csv('F:/VVIP/VVIP_CELL.csv')
#full_cellname = DataFrame(columns=['cellname'])
#full_cellname['cellname'] = full_cellname1['CellName']


cell_net = pd.read_csv('F:/work/cell_200net_new.csv')
cell_net_need = cell_net[['Description', 'CellName']]
cell_net_need.columns = ['net_num', 'cellname']

date_lost_list = wanzhengxing(zhibiao_dropNa_ca, full_cellname)


date_lost_list_netnum = pd.merge(date_lost_list, cell_net_need, on='cellname', how='left')


zhibiao_full_last = data_full(zhibiao_dropNa_ca, date_lost_list_netnum)
#zhibiao_full_last1 = zhibiao_full_last.drop([zhibiao_full_last['net_num']=='0.0'])






#计算工作日和假日的流量中值
flow_full_last = zhibiao_full_last[['time','cellname','flow','hour']]
canlender_2018 = pd.read_csv('F:/VVIP/2018workday.csv')

#workday_calender = DataFrame(list(canlender_2018), columns=['date'])
workday_2018 = str2time(list(canlender_2018['date'])) 
workday_2018_list = list()
for date_item in workday_2018:
    workday_2018_list.append(date_item[:10])

zhibiao_full_last_timelist = flow_full_last['time']

zhibiao_full_last_timelist_days = list()
for time_full_item in zhibiao_full_last_timelist:
    zhibiao_full_last_timelist_days.append(time_full_item[:10])
    
flow_full_last.index = zhibiao_full_last_timelist_days
flow_full_date_list = list(set(flow_full_last.index))

workday_date_list = list()
holiday_date_list = list()

for workday_item in flow_full_date_list:
    if workday_item in workday_2018_list:
        workday_date_list.append(workday_item)
    else:
        holiday_date_list.append(workday_item)

workday_data = flow_full_last.ix[workday_date_list]
workday_data['flow'] = workday_data['flow'].astype(float)
workday_cell_median = workday_data.groupby(['cellname','hour'])['flow'].median()/1024

holiday_data = flow_full_last.ix[holiday_date_list]
holiday_data['flow'] = holiday_data['flow'].astype(float)
holiday_cell_median = holiday_data.groupby(['cellname','hour'])['flow'].median()/1024



'''
workday_cell_sum = workday_data.groupby(['cellname','hour'])['flow'].sum()
workday_cell_count = workday_data.groupby(['cellname','hour'])['flow'].count()
workday_cell_median = workday_cell_sum/workday_cell_count/1024
workday_cell_median.columns = 'flow'
'''

workday_median_path = pwd + 'workday_cell_median.csv'
if os.path.exists(workday_median_path):
    os.remove(workday_median_path)
workday_cell_median.to_csv(workday_median_path)

'''
holiday_cell_sum = holiday_data.groupby(['cellname','hour'])['flow'].sum()
holiday_cell_count = holiday_data.groupby(['cellname','hour'])['flow'].count()
holiday_cell_median = holiday_cell_sum/holiday_cell_count/1024
holiday_cell_median.columns = 'flow'
'''

holiday_median_path = pwd + 'holiday_cell_median.csv'
if os.path.exists(holiday_median_path):
    os.remove(holiday_median_path)
holiday_cell_median.to_csv(holiday_median_path)




#zhibiao_full_last_group = zhibiao_full_last.groupby(['cellname', ])

zhibiao_grouped = zhibiao_full_last.groupby(['net_num', 'year', 'month', 'day', 'hour'])
#zhibiao_grouped = zhibiao_full_last1.groupby(['net_num', 'year', 'month', 'day', 'hour'])

zhibiao_flow = zhibiao_grouped['flow'].sum()
zhibiao_rrc = zhibiao_grouped['rrc'].sum()
zhibiao_erab = zhibiao_grouped['erab'].sum()
zhibiao_handover = zhibiao_grouped['handover'].sum()
zhibiao_ue = zhibiao_grouped['ue'].sum()
#新增4个指标
zhibiao_full_last['tcp'] = zhibiao_full_last['tcp'].astype(float)
zhibiao_full_last['volte'] = zhibiao_full_last['volte'].astype(float)
zhibiao_full_last['big_rate'] = zhibiao_full_last['big_rate'].astype(float)
zhibiao_full_last['small_delay'] = zhibiao_full_last['small_delay'].astype(float)


zhibiao_tcp = zhibiao_grouped['tcp'].median()
zhibiao_volte = zhibiao_grouped['volte'].median()
zhibiao_big_rate = zhibiao_grouped['big_rate'].median()
zhibiao_small_delay = zhibiao_grouped['small_delay'].median()

zhibiao_flow = zhibiao_flow/1024**2
zhibiao_rrc = zhibiao_rrc/1024
zhibiao_erab = zhibiao_erab/1024
zhibiao_handover = zhibiao_handover/1024
zhibiao_ue = zhibiao_ue/10

result = pd.DataFrame({'flow':zhibiao_flow, 'erab':zhibiao_erab, 'UE':zhibiao_ue, 
                       'handover':zhibiao_handover, 'rrc':zhibiao_rrc, 'tcp':zhibiao_tcp, 
                       'volte':zhibiao_volte, 'big_rate':zhibiao_big_rate, 'small_delay':zhibiao_small_delay})

columns_list_last = ['UE', 'erab', 'flow','handover', 'rrc', 'tcp', 'volte', 'big_rate', 'small_delay']
result = result.ix[:, columns_list_last]
    
result.to_csv('F:/VVIP/cell_mean/result.csv')
    
    
'''
    net_list = list()
for i in range(len(result.index)) :
    net_list.append(result.index[i][0])
net_list = list(set(net_list))  
stand_times = 0 

out = standard_data(result, net_list, stand_times)

 '''   







#zhibiao_liuliang = zhibiao_liuliang/1024**2

#zhibiao_liuliang.to_csv('F:/work/tianhe4location/zhibiao_liuliang.csv')