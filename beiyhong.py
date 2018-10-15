# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:20:22 2018

@author: User
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import get_rNB as get

#判断两个dataframe的值是否相等，返回一个包含True，和False的mask DataFrame
def data_judge(grouped1, grouped2):
    grouped_judged = pd.DataFrame(index=grouped1.index, columns=grouped1.columns)
    net_index = grouped1.index
    for net in net_index:
        grouped_judged.ix[net] = grouped1.ix[net].values == grouped2.ix[net].values
    return grouped_judged

#分别计算两个标准值
def data_devide(grouped_static, day_div):
    workday_standard = list()
    weekend_standard = list()
    for grouped in grouped_static:
        workday_standard.append(grouped[day_div[0]])
        weekend_standard.append(grouped[day_div[1]])
        #workday_standard.append(grouped[day_div[0]].mean(axis=1))
        #weekend_standard.append(grouped[day_div[1]].mean(axis=1))
    return workday_standard, weekend_standard

#timestamp 转换为字符串listl
def time_stamp_to_list(time_stamp):
    timelist = []
    for stamp in time_stamp:
        timelist.append(stamp.strftime('%Y-%m-%d %H:%M:%S.000'))
    return timelist

#取周末和工作日数据
def Bday_Weekend(date_start, date_end):
    time_columns = pd.date_range(date_start,  date_end)
    #计算工作时间
    time_columns_BD = pd.date_range(date_start,  date_end, freq='B')
    time_bd = time_stamp_to_list(time_columns_BD)
    #计算周末时间
    time_columns_weekend = time_columns.difference(time_columns_BD)
    time_weekend = time_stamp_to_list(time_columns_weekend)
    #返回两组时间
    time_div = list([time_bd, time_weekend])
    return time_div

#按时间和网格进行聚合得到不同的网格统计值
def net_converge_mask(Indataframe, data_mask): 
    
    grouped = Indataframe.groupby(['net_num', '开始时间'])
   
    #计算流量平均值
    grouped_sum = grouped['MR-总流量（KB）'].sum().unstack()
    #grouped_sum = grouped_traffic_sum[data_mask]
    #计算最大值
    grouped_max = grouped['MR-总流量（KB）'].max().unstack()
    #grouped_max = grouped_traffic_max[data_mask]
    #计算最小值
    grouped_min = grouped['MR-总流量（KB）'].min().unstack()
    #grouped_min = grouped_traffic_min[data_mask]
    #计算0.75分位值
    grouped_q43 = grouped['MR-总流量（KB）'].quantile(0.75).unstack()
    #grouped_43 = grouped_traffic_q43[data_mask]
    #计算中位数
    grouped_median = grouped['MR-总流量（KB）'].median().unstack()
    #计算网格小区数
    grouped_netnum = grouped['小区名称'].count().unstack()
    #计算网格各小区流量变化率加权和
    grouped_net_cellratio = grouped['cell_ratio'].sum().unstack()
    
    grouped_net_cellratio_new = grouped_net_cellratio/grouped_sum
    #计算均方差
   #grouped_traffic_std = grouped['MR-总流量（KB）'].std().unstack()
   #grouped_std = grouped_traffic_std[data_mask]
    #计算方差
   #grouped_traffic_var = grouped['MR-总流量（KB）'].var().unstack()
   #grouped_var = grouped_traffic_var[data_mask]
    #返回结果
    grouped_static = list([grouped_sum, grouped_max, grouped_min, grouped_q43, grouped_median, grouped_netnum, grouped_net_cellratio_new])#grouped_std, grouped_var])
    return grouped_static

#按时间和网格进行聚合得到不同的网格统计值
def net_converge(Indataframe): 
    grouped_static = list()
    grouped = Indataframe.groupby(['net_num', '开始时间'])
    #计算流量平均值
    grouped_traffic_sum = grouped['MR-总流量（KB）'].sum().unstack()
    #计算最大值
    grouped_traffic_max = grouped['MR-总流量（KB）'].max().unstack()
    #计算最小值
    grouped_traffic_min = grouped['MR-总流量（KB）'].min().unstack()
    #计算0.75分位值
    grouped_traffic_q43 = grouped['MR-总流量（KB）'].quantile(0.75).unstack()
    #计算中位数
    grouped_traffic_median = grouped['MR-总流量（KB）'].median().unstack()
    
    #计算网格小区数
    grouped_traffic_netnum = grouped['小区名称'].count().unstack()
    
    #计算网格各小区流量变化率加权和
    grouped_net_cellratio = grouped['cell_ratio'].sum().unstack()
    
    grouped_net_cellratio_new = grouped_net_cellratio/grouped_traffic_sum
    #计算均方差
   #grouped_traffic_std = grouped['MR-总流量（KB）'].std().unstack()
    #计算方差
   #grouped_traffic_var = grouped['MR-总流量（KB）'].var().unstack()
    #返回结果
    grouped_static = list([grouped_traffic_sum, grouped_traffic_max, 
                           grouped_traffic_min, grouped_traffic_q43, 
                           grouped_traffic_median, grouped_traffic_netnum,
                           grouped_net_cellratio_new])#, grouped_traffic_var])
    return grouped_static
#计算各网格的基准值，包括均值和方差
def standard_static(instatic, day_num):
    outstatic_median = list()
    outstatic_std = list()
    for statics in instatic:
        statics = statics.dropna(thresh=(day_num*3//4))
        outstatic_median.append(statics.median(axis=1))
        outstatic_std.append(statics.std(axis=1))
    return outstatic_median, outstatic_std
 
#组建标准特征(取特征并集，去掉特征不完整的网格)
def standard_clear(inlist):
    index_list = list()
    for statics in inlist:
        index_list.append(statics.index)
    index_out = set(index_list[0])
    for indexs in index_list[1:]:
        index_out = index_out & set(indexs)
    outlist = list()
    for statics in inlist:
        statics = statics[index_out]
        outlist.append(statics)
    return outlist    

#计算特征向量       
def static_vector(inlist):
    net_index = inlist[0].index
    in_array = inlist[0].values.reshape(len(inlist[0]),1)
    for lists in inlist[1:]:
        in_array = np.hstack((in_array,lists.values.reshape(len(lists),1)))
    out_vector = DataFrame(in_array)
    out_vector.index = net_index
    if len(inlist) == 7:
        out_vector.columns = ['sum', 'max', 'min', 'q43', 'median', 'netnum', 'net_cellratio']#td', 'var']
    else:
        out_vector.columns = ['sum_ratio', 'sum_sub', 'sum', 'max', 'min', 'q43', 'median', 'netnum', 'net_cellratio']
    return out_vector

#将数组还原格式
def grouped_stack(grouped):
    stack_list = list()
    for groupeds in grouped:
        stack_list.append(groupeds.stack())
    return stack_list

#计算差值得到样本值和基准值的差值和百分比
def sample_cal(data_vector, standard_vector):
    out_list = list()
    data_vector = data_vector.unstack()
    vector_columns = list(standard_vector.columns)
    for columns in vector_columns:
        if columns == 'sum':
            data_columns = data_vector[columns]
            #index_old = data_columns.index
            #columns_old = data_columns.columns
            #data_columns = np.array(data_columns)
            columns_standard = standard_vector[columns]
            #aa = len(columns_standard)
            #columns_standard = np.array(columns_standard)
            #columns_standard = columns_standard.reshape((aa, 1))
            data_columns = data_columns.div(columns_standard, axis=0)
            #data_columns = DataFrame(data_columns)
            #data_columns.index = index_old
            #data_columns.columns = columns_old
            out_list.append(data_columns)
            out_list.append(data_vector[columns].sub(columns_standard, axis=0))
            out_list.append(data_vector[columns])
        elif columns != 'netnum' and columns != 'net_cellratio':
            data_columns = data_vector[columns]
            #index_old = data_columns.index
            #columns_old = data_columns.columns
            #data_columns = np.array(data_columns)
            columns_standard = standard_vector[columns]
            #aa = len(columns_standard)
            #columns_standard = np.array(columns_standard)
            #columns_standard = columns_standard.reshape((aa, 1))
            data_columns = data_columns.sub(columns_standard, axis=0)
            #data_columns = DataFrame(data_columns)
            #data_columns.index = index_old
            #data_columns.columns = columns_old
            out_list.append(data_columns)
        else:
            data_columns = data_vector[columns]
            out_list.append(data_columns)
            
    out_list = grouped_stack(out_list)
    out_list = static_vector(out_list)
    #计算欧式距离
    out_list['EDistance'] = out_list.apply(lambda x: np.sqrt(x['max']**2 + x['min']**2 +x['q43']**2 +x['median']**2), axis=1)
    return out_list

#计算欧式距离的比值
def EDistance_ratio(in_vector):
    ED = in_vector['EDistance']
    ed = ED.unstack().median(axis=1)
    ed_ratio = ED.div(ed)
    in_vector['EDistance_ratio'] = ed_ratio
    return in_vector

#计算最大列和最大行
def max_col_row(inlist):
    letter_dic = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10, 
              'K':11, 'L':12, 'M':13, 'N':14, 'O':15, 'P':16, 'Q':17, 'R':18, 'S':19, 'T':20, 
              'U':21, 'V':22, 'W':23, 'X':24, 'Y':25, 'Z':26}
    cols_list = list()
    rows_list = list()
    for member in inlist:
        member_list = list(member)
        columns_letter = list()
        rows_letter = list()
       #得到字符串的字母部分和数字部分
        for letter in member_list:
            if letter in letter_dic:
                columns_letter.append(letter)
            else:
                rows_letter.append(letter)         
        columns_letter_str = "".join(columns_letter)
        rows_letter_str = "".join(rows_letter)
        cols_list.append(columns_letter_str)
        cols_list.sort()
        rows_list.append(int(rows_letter_str))
        rows_list.sort()
    max_columns = cols_list[len(cols_list)-1]
    max_rows = rows_list[len(rows_list)-1]
    return max_columns, max_rows
        
#去重复项
def cancel_dup(input):
    output = list(set(input))
    return output

#计算邻网格
def net_NB(y):
    max_columns, max_rows = max_col_row(y)
    NB_list = list()
    get_nb = get.get_NB(max_columns, max_rows)
    for net in y:
        NB_list.append(get_nb.net_neighbor(net))
    NB_list = DataFrame(NB_list)
    NB_list.index = y
    return NB_list
    

#读取数据
zhibiao = pd.read_csv('D:/data/500/new/data_12.csv')
zhibiao = zhibiao[['开始时间', '小区名称', 'MR-总流量（KB）']]
cell_500net = pd.read_csv('D:/data/500/new/cell_500net.csv')
cell_500net = cell_500net[['Description', 'CellName']]
cell_500net.columns = [['net_num', '小区名称']]

#merge数据
zhibiao_net = pd.merge(zhibiao, cell_500net, on='小区名称')

#数据去NaN ，去零值

#zhibiao_net_dropNa_clear = zhibiao_net.dropna()
zhibiao_net_dropNa = zhibiao_net.dropna()
zhibiao_net_clear = np.where(zhibiao_net_dropNa['MR-总流量（KB）']==0)
zero_row = list(zhibiao_net_clear[0])
zhibiao_net_dropNa_clear = zhibiao_net_dropNa.drop(zero_row)

#取网格的小区数
net_cell_num = cell_500net.groupby('net_num').count()
net_cell_num.columns = ['cell_num']

#取按时间和网格的小区计数
grouped_cellnum = zhibiao_net_dropNa_clear.groupby(['net_num', '开始时间'])
grouped_cellnum_new = grouped_cellnum['小区名称'].count().unstack()

#得有效数据判定结果
data_judge_mask = data_judge(grouped_cellnum_new, net_cell_num)

#计算统计量
grouped_static_mask = net_converge_mask(zhibiao_new, data_judge_mask)
#工作日和周末划分wor
day_div = Bday_Weekend('2017-12-1', '2017-12-31')
workday_num = len(day_div[0])
weekend_num = len(day_div[1])

#分别计算工作日和非工作日的统计指标

workday, weekend = data_devide(grouped_static_mask, day_div)  

#计算得到各网格的基准值包括均值和方差
workday_median_out, workday_std_out = standard_static(workday, workday_num)
weekend_median_out, weekend_std_out = standard_static(weekend, weekend_num)

#将多特征求并集得到完整特征
workday_median = standard_clear(workday_median_out) 
workday_std = standard_clear(workday_std_out)
weekend_median = standard_clear(weekend_median_out)
weekend_std = standard_clear(weekend_std_out)

#计算得到特征向量
workday_median_vector = static_vector(workday_median)
workday_std_vector = static_vector(workday_std)
weekend_median_vector = static_vector(weekend_median)
weekend_std_vector = static_vector(weekend_std)

#计算各网格的各种统计量
grouped_static = net_converge(zhibiao_new)
workday_grouped_data, weekend_grouped_data = data_devide(grouped_static, day_div)

#将数据格式还原
workday_data_stack = grouped_stack(workday_grouped_data)
weekend_data_stack = grouped_stack(weekend_grouped_data)

#计算向量
workday_data_vector = static_vector(workday_data_stack)
weekend_data_vector = static_vector(weekend_data_stack)

#取有效数据
index_workday_vector = workday_median_vector.index
index_weekend_vector = weekend_median_vector.index

workday_data_vector_new = workday_data_vector.ix[index_workday_vector]
weekend_data_vector_new = weekend_data_vector.ix[index_weekend_vector]

#计算得到向量数据
workday_out = sample_cal(workday_data_vector_new, workday_median_vector)
weekend_out = sample_cal(weekend_data_vector_new, weekend_median_vector)

#计算欧式距离
    
workday_out_ED = EDistance_ratio(workday_out)
weekend_out_ED = EDistance_ratio(weekend_out)

#合并工作日和周末的向量表
data_vector = pd.merge(workday_out_ED, weekend_out_ED,
                       on=['sum_ratio', 'sum_sub', 'sum', 'max', 'min', 'q43',
                       'median','netnum', 'net_cellratio', 'EDistance', 'EDistance_ratio'], 
                           left_index=True, right_index=True, how='outer')



#提取网格列表并求得其邻网格列表
#data_vector_index = data_vector.unstack().index
#index_net_neighbor = net_NB(data_vector_index)

#计算各网格周边网格的sum_ratio和sum_sub的情况                                      
#data_vector_sumratio = data_vector['sum_ratio']
#data_vector_sumsub = data_vector['sum_sub']

#NB_sumratio = data_vector_sumratio.unstack()
#NB_sumsub = data_vector_sumsub.unstack()

#计算各网格的周边网格的sumratio 最大值，最小值， sum_sub的最大值，最小值
#def NB_zhibiao(net_number, NB_sumratio, NB_sumsub):
#    index_net_neighbor = net_NB(data_vector_index)
#    net_neighbor_list = index_net_neighbor.ix[net_number]
    
#    list_sumratio = NB_sumratio.ix[net_neighbor_list]    
#    list_sumratio_max = list_sumratio.max(axis=0)
#    list_sumratio_min = list_sumratio.min(axis=0) 
    
#    list_sumsub = NB_sumsub.ix[net_neighbor_list]    
#    list_sumsub_max = list_sumsub.max(axis=0)
#    list_sumsub_min = list_sumsub.min(axis=0) 
    
#    NB_zhibiao_out = DataFrame({'ratio_max':list_sumratio_max, 'ratio_min':list_sumratio_min,
#                                'sumsub_max':list_sumsub_max, 'sumsub_min':list_sumsub_min})
    
#    return NB_zhibiao_out
#计算各小区的标准流量及每天各小区的流量变化比例
#分工作日非工作日计算基准：
def cell_ratio(zhibiao_net_dropNa_clear):
    zhibiao_net_dropNa_clear.index = zhibiao_net_dropNa_clear['开始时间']
    day_div = day_div = Bday_Weekend('2017-12-1', '2017-12-31')
    workday_zhibiao = zhibiao_net_dropNa_clear.loc[day_div[0]]
    workday_zhibiao.index = workday_zhibiao['小区名称']
    weekend_zhibiao = zhibiao_net_dropNa_clear.loc[day_div[1]]
    weekend_zhibiao.index = weekend_zhibiao['小区名称']

    workday_cell_group = workday_zhibiao.groupby(['小区名称'])
    workday_cell_standard = workday_cell_group['MR-总流量（KB）'].median()
    workday_cell_standard = DataFrame(workday_cell_standard)

    weekend_cell_group = weekend_zhibiao.groupby(['小区名称'])
    weekend_cell_standard = weekend_cell_group['MR-总流量（KB）'].median()
    weekend_cell_standard = DataFrame(weekend_cell_standard)


    workday_zhibiao_new = pd.merge(workday_zhibiao, workday_cell_standard,
                                        how='outer', left_index=True, right_index=True)
    workday_zhibiao_new.columns = ['开始时间', '小区名称', 'MR-总流量（KB）', 'net_num', '月标准流量']

    weekend_zhibiao_new = pd.merge(weekend_zhibiao, weekend_cell_standard,
                                        how='outer', left_index=True, right_index=True)
    weekend_zhibiao_new.columns = ['开始时间', '小区名称', 'MR-总流量（KB）', 'net_num', '月标准流量']

    zhibiao_new = pd.concat([workday_zhibiao_new,weekend_zhibiao_new],axis=0)
    zhibiao_new['cell_ratio'] = zhibiao_new['MR-总流量（KB）']**2/zhibiao_new['月标准流量']

    return zhibiao_new
