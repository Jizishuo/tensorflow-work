# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:56:06 2018

@author: 陆庆国

version: 3.0
"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import get_NB as get

class get_vector(object): 
    
    def __init__(self, start, end):
        self.__letter_dic = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10, 
              'K':11, 'L':12, 'M':13, 'N':14, 'O':15, 'P':16, 'Q':17, 'R':18, 'S':19, 'T':20, 
              'U':21, 'V':22, 'W':23, 'X':24, 'Y':25, 'Z':26}
        self.__start = start
        self.__end = end
        
    #判断两个dataframe的值是否相等，返回一个包含True，和False的mask DataFrame
    def __data_judge(self, grouped1, grouped2):
        grouped_judged = pd.DataFrame(index=grouped1.index, columns=grouped1.columns)
        net_index = grouped1.index
        for net in net_index:
            grouped_judged.ix[net] = grouped1.ix[net].values == grouped2.ix[net].values
        return grouped_judged

    #分别计算两个标准值
    def __data_devide(self, grouped_static, day_div):
        workday_standard = list()
        weekend_standard = list()
        for grouped in grouped_static:
            workday_standard.append(grouped[day_div[0]])
            weekend_standard.append(grouped[day_div[1]])
        #workday_standard.append(grouped[day_div[0]].mean(axis=1))
        #weekend_standard.append(grouped[day_div[1]].mean(axis=1))
        return workday_standard, weekend_standard

    #timestamp 转换为字符串listl
    def __time_stamp_to_list(self, time_stamp):
        timelist = []
        for stamp in time_stamp:
            timelist.append(stamp.strftime('%Y-%m-%d %H:%M:%S.000'))
        return timelist

    #取周末和工作日数据
    def __Bday_Weekend(self, date_start, date_end):
        time_columns = pd.date_range(date_start,  date_end)
        #计算工作时间
        time_columns_BD = pd.date_range(date_start,  date_end, freq='B')
        time_bd = self.__time_stamp_to_list(time_columns_BD)
        #计算周末时间
        time_columns_weekend = time_columns.difference(time_columns_BD)
        time_weekend = self.__time_stamp_to_list(time_columns_weekend)
        #返回两组时间
        time_div = list([time_bd, time_weekend])
        return time_div

    #按时间和网格进行聚合得到不同的网格统计值
    def __net_converge_mask(self, Indataframe):#, data_mask): 
        grouped_static = list()
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
        #计算网格中的小区数
       # grouped_netnum = grouped['小区名称'].count().unstack
        #grouped_median = grouped_traffic_median[data_mask]
        #计算均方差
        #grouped_traffic_std = grouped['MR-总流量（KB）'].std().unstack()
        #grouped_std = grouped_traffic_std[data_mask]
        #计算方差
        #grouped_traffic_var = grouped['MR-总流量（KB）'].var().unstack()
        #grouped_var = grouped_traffic_var[data_mask]
        #返回结果
        grouped_static = list([grouped_sum, grouped_max, grouped_min, grouped_q43, 
                               grouped_median, grouped_netnum])#, grouped_netnum])#grouped_std, grouped_var])
        return grouped_static

    #按时间和网格进行聚合得到不同的网格统计值
    def __net_converge(self, Indataframe): 
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
        #计算网格中的小区数
        grouped_traffic_netnum = grouped['MR-总流量（KB）'].count().unstack()
        #计算均方差
        #grouped_traffic_std = grouped['MR-总流量（KB）'].std().unstack()
        #计算方差
        #grouped_traffic_var = grouped['MR-总流量（KB）'].var().unstack()
        #返回结果
        grouped_static = list([grouped_traffic_sum, grouped_traffic_max, 
                           grouped_traffic_min, grouped_traffic_q43, 
                           grouped_traffic_median, grouped_traffic_netnum]), #grouped_traffic_netnum])#grouped_traffic_std, grouped_traffic_var])
        return grouped_static
    #计算各网格的基准值，包括均值和方差
    def __standard_static(self, instatic, day_num):
        outstatic_median = list()
        outstatic_std = list()
        for statics in instatic:
            statics = statics.dropna(thresh=(day_num*3//4))
            outstatic_median.append(statics.median(axis=1))
            outstatic_std.append(statics.std(axis=1))
        return outstatic_median, outstatic_std
 
    #组建标准特征(取特征并集，去掉特征不完整的网格)
    def __standard_clear(self, inlist):
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
    def __static_vector(self, inlist):
        net_index = inlist[0].index
        in_array = inlist[0].values.reshape(len(inlist[0]),1)
        for lists in inlist[1:]:
            in_array = np.hstack((in_array,lists.values.reshape(len(lists),1)))
        out_vector = DataFrame(in_array)
        out_vector.index = net_index
        if len(inlist) == 6:
            out_vector.columns = ['sum', 'max', 'min', 'q43', 'median', 'netnum']#td', 'var']
        else:
            out_vector.columns = ['sum_ratio', 'sum_sub', 'sum', 'max', 'min',
                                  'q43', 'median', 'netnum']#, 'netnum']
        return out_vector

    #将数组还原格式
    def __grouped_stack(self, grouped):
        stack_list = list()
        for groupeds in grouped:
            stack_list.append(groupeds.stack())
        return stack_list

    #计算差值得到样本值和基准值的差值和百分比
    def __sample_cal(self, data_vector, standard_vector):
        out_list = list()
        #    data_index = standard_vecor.index
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
            elif columns != 'netnum':
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
        out_list = self.__grouped_stack(out_list)
        out_list = self.__static_vector(out_list)
        #计算欧式距离
        out_list['EDistance'] = out_list.apply(lambda x: np.sqrt(x['max']**2 + x['min']**2 
                +x['q43']**2 +x['median']**2), axis=1)
        return out_list

    #计算欧式距离的比值
    def __EDistance_ratio(self, in_vector):
        ED = in_vector['EDistance']
        ed = ED.unstack().median(axis=1)
        ed_ratio = ED.div(ed)
        in_vector['EDistance_ratio'] = ed_ratio
        return in_vector

    #计算最大列和最大行
    def __max_col_row(self, inlist):
        cols_list = list()
        rows_list = list()
        for member in inlist:
            member_list = list(member)
            columns_letter = list()
            rows_letter = list()
       #得到字符串的字母部分和数字部分
            for letter in member_list:
                if letter in self.__letter_dic:
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
    def __cancel_dup(self, input):
        output = list(set(input))
        return output

    #得到日期列表
    def __date_index(self):
        start = self.__start
        end = self.__end
        date_index = pd.date_range(start,  end)
        out_data_index = self.__time_stamp_to_list(date_index)
        return out_data_index
    
    #计算邻网格
    def __net_NB(self, y):
        max_columns, max_rows = self.__max_col_row(y)
        NB_list = list()
        get_nb = get.get_NB(max_columns, max_rows)
        for net in y:
            NB_list.append(get_nb.net_neighbor(net))
        NB_list = DataFrame(NB_list)
        NB_list.index = y
        
        #计算日期index
        date_index = self.__date_index()
        return NB_list, date_index

    #数据预处理    
    def indata_handle(self, indata, in_cell_net):
        in_zhibiao = indata[['开始时间', '小区名称', 'MR-总流量（KB）']]
        cell_net = in_cell_net[['Description', 'CellName']]
        cell_net.columns = [['net_num', '小区名称']]
        #按小区名称merge，合并指标和网格
        zhibiao_net = pd.merge(in_zhibiao, cell_net, on='小区名称')
        #数据去邻值，去NaN值
        zhibiao_net_dropNa = zhibiao_net.dropna()
        zhibiao_net_clear = np.where(zhibiao_net_dropNa['MR-总流量（KB）']==0)
        zero_row = list(zhibiao_net_clear[0])
        zhibiao_net_dropNa_clear = zhibiao_net_dropNa.drop(zero_row)
        #计算统计量
        grouped_static = self.__net_converge_mask(zhibiao_net_dropNa_clear)
        #,data_judge_mask)
        return grouped_static

    def standard_vector(self, in_zhibiao):
        #工作日和周末划分wor
        #grouped_static_mask = net_converge_mask(in_zhibiao)
        day_div = self.__Bday_Weekend(self.__start, self.__end)
        workday_num = len(day_div[0])
        weekend_num = len(day_div[1])
        #分别计算工作日和非工作日的统计指标
        workday, weekend = self.__data_devide(in_zhibiao, day_div) 
        #计算得到各网格的基准值包括均值和方差
        workday_median_out, workday_std_out = self.__standard_static(workday, workday_num)
        weekend_median_out, weekend_std_out = self.__standard_static(weekend, weekend_num)
        #将多特征求并集得到完整特征
        workday_median = self.__standard_clear(workday_median_out) 
        #workday_std = standard_clear(workday_std_out)
        weekend_median = self.__standard_clear(weekend_median_out)
        #weekend_std = standard_clear(weekend_std_out)
        #计算得到特征向量
        workday_median_vector = self.__static_vector(workday_median)
        #workday_std_vector = static_vector(workday_std)
        weekend_median_vector = self.__static_vector(weekend_median)
        #weekend_std_vector = static_vector(weekend_std)
        return workday_median_vector, weekend_median_vector

    #计算每天的统计量
    def data_vector(self, input_zhibiao):
        day_div = self.__Bday_Weekend(self.__start, self.__end)
        #workday_num = len(day_div[0])
        #weekend_num = len(day_div[1])
        #grouped_static = net_converge(input_zhibiao)
        workday_grouped_data, weekend_grouped_data = self.__data_devide(input_zhibiao, day_div)

        #将数据格式还原
        workday_data_stack = self.__grouped_stack(workday_grouped_data)
        weekend_data_stack = self.__grouped_stack(weekend_grouped_data)

        #计算向量
        workday_data_vector = self.__static_vector(workday_data_stack)
        weekend_data_vector = self.__static_vector(weekend_data_stack)
        return workday_data_vector, weekend_data_vector

    #利用标准向量和data向量计算输出指标向量
    def cal_data_vector(self, in_data_vector, in_standard_vector):
        #取有效数据
        index_standard_vector = in_standard_vector.index
        #index_weekend_vector = weekend_median_vector.index

        in_data_vector_new = in_data_vector.ix[index_standard_vector]
        #weekend_data_vector_new = weekend_data_vector.ix[index_weekend_vector]

        #计算得到向量数据
        workday_out = self.__sample_cal(in_data_vector_new, in_standard_vector)
        #weekend_out = sample_cal(weekend_data_vector_new, weekend_median_vector)
    
        #计算并计算欧式距离
        out_ED = self.__EDistance_ratio(workday_out)
        #weekend_out_ED = EDistance_ratio(weekend_out)
        return out_ED
    
    #得到输出指标向量  
    def out_data_vector(self, work_ED, weekend_ED) :
        #合并工作日和周末的向量表
        out_data_vector = pd.merge(work_ED, weekend_ED,
                       on=['sum_ratio', 'sum_sub', 'sum', 'max', 'min', 'q43',
                       'median', 'netnum','EDistance', 'EDistance_ratio'], 
                           left_index=True, right_index=True, how='outer')
        return out_data_vector



    def out_data_vector_neighbor(self, inlist):
        #得到需要处理的网格列表
        idx = pd.IndexSlice
        net_index = list(inlist.unstack().index)
        #计算得到需处理的网格邻网格列表，同时得到一个日期列表
        net_neighbor, date_index = self.__net_NB(net_index)
        
        #计算sum_ratio相关指标
        sumratio_unstack = inlist['sum_ratio'].unstack()
        sumsub_unstack = inlist['sum_sub'].unstack()
        comp_number_unstack = inlist['comp_number'].unstack()
        history_comp_number_umstack = inlist['history_comp_number'].unstack()
        sumratio_min = list()
        sumratio_median = list()
        sumratio_min_net = list()
        sumratio_min_sumsub = list()
        sumsub_sum = list()
        comp_number = list()
        history_comp_number = list()
        #计算并得到邻网格的最小值和中位值
        for net in net_index:
            neighbor_list = list(net_neighbor.ix[net])
            ret_list = list(set(neighbor_list)&set(net_index))
            #计算sumratio相关指标
            sumratio_static_list = sumratio_unstack.loc[idx[ret_list], :]
            sumsub_static_list = sumsub_unstack.loc[idx[ret_list], :]
            
            sumratio_static_min = sumratio_unstack.loc[idx[ret_list], :].min(axis=0)
            sumratio_static_median = sumratio_unstack.loc[idx[ret_list], :].median(axis=0)
            sumratio_static_min_netname = self.__find_min_net(sumratio_static_list, sumratio_static_min)
            sumsub_static_min_subsum = self.__find_min_net_sumsub(sumsub_static_list, sumratio_static_min_netname, sumratio_static_min)
            sumsub_static_sum = sumsub_unstack.loc[idx[ret_list], :].sum(axis=0)
            comp_static_number = comp_number_unstack.loc[idx[ret_list], :].sum(axis=0)
            history_comp_static_number = history_comp_number_umstack.loc[idx[ret_list], :].sum(axis=0)
            
            sumratio_min.append(sumratio_static_min)
            sumratio_median.append(sumratio_static_median)
            sumratio_min_net.append(sumratio_static_min_netname)
            sumratio_min_sumsub.append(sumsub_static_min_subsum)
            sumsub_sum.append(sumsub_static_sum)
            comp_number.append(comp_static_number)
            history_comp_number.append(history_comp_static_number)
       
        result_min = DataFrame(sumratio_min)
        result_min.index = net_index
        result_min = result_min.fillna(0)
        result_min = result_min.stack()
        result_min_index = result_min.unstack().index
                                     
        result_median = DataFrame(sumratio_median)
        result_median.index = net_index
        result_median = result_median.fillna(0)
        result_median = result_median.stack()
        
        result_min_netname = DataFrame(sumratio_min_net)
        result_min_netname.index = net_index
        #result_min_netname.columns = date_index
        result_min_netname = result_min_netname.fillna(0)
        result_min_netname = result_min_netname.stack()
        
        result_min_sumsub = DataFrame(sumratio_min_sumsub)
        result_min_sumsub.index = net_index
        #result_min_sumsub.columns = date_index
        result_min_sumsub = result_min_sumsub.fillna(0)
        result_min_sumsub = result_min_sumsub.stack()
        
        result_sumsub_sum = DataFrame(sumsub_sum)
        result_sumsub_sum.index = net_index
        result_sumsub_sum = result_sumsub_sum.fillna(0)
        result_sumsub_sum = result_sumsub_sum.stack()
        
        result_comp_number = DataFrame(comp_number)
        result_comp_number.index = net_index
        result_comp_number = result_comp_number.fillna(0)
        result_comp_number = result_comp_number.stack()
        
        result_history_comp_number = DataFrame(history_comp_number)
        result_history_comp_number.index = net_index
        result_history_comp_number = result_history_comp_number.fillna(0)
        result_history_comp_number = result_history_comp_number.stack()
        
        inlist['neighbor_min'] = result_min
        inlist['neighbor_median'] = result_median
        inlist['neighbor_min_netname'] = result_min_netname
        inlist['neighbor_min_sumsub'] = result_min_sumsub
        inlist['neighbor_sumsub'] = result_sumsub_sum
        inlist['neighbor_comp_num'] = result_comp_number   
        inlist['neighbor_history_comp_num'] = result_history_comp_number
              
        #result_median_index = result_median.unstack().index
        #将计算得到的邻网格的最小值和平均值写入数据向量中                           
        #for net in result_min_index:
            #for date_ID in date_index:
                #inlist.loc[idx[net, date_ID], 'neighbor_min'] = result_min.loc[idx[net, date_ID], ]
                       
                #inlist.loc[idx[net, date_ID], 'neighbor_median'] = result_median.loc[idx[net, date_ID], ]
                
                #inlist.loc[idx[net, date_ID], 'neighbor_min_netname'] = result_min_netname.loc[idx[net, date_ID], ]
                       
                #inlist.loc[idx[net, date_ID], 'neighbor_min_sumsub'] = result_min_sumsub.loc[idx[net, date_ID], ]
                       
                #inlist.loc[idx[net, date_ID], 'neighbor_sumsub'] = result_sumsub_sum.loc[idx[net, date_ID], ]
        
        return inlist



    #计算sum_ratio最小值网格名称
    def __find_min_net(self, inlist, reflist):
        if inlist.shape[0] != 0:
            min_net_name = list()
            inlist = inlist.fillna(100)
            inlist_columns = inlist.columns
            inlist_index = inlist.index
            for date_number in inlist_columns:
                #zhibiao_date = list(inlist[date_number])
                zhibiao_min = inlist[date_number].min()
                #zhibiao_min_index = inlist[date_number].index
                min_location = list(inlist[date_number]).index(zhibiao_min)
                min_netname = inlist_index[min_location]
                min_net_name.append(min_netname)
            min_net_name = Series(min_net_name, index=inlist_columns)
            return min_net_name
        else:
            return reflist

        #计算sumratio最小网格的sumsub
    def __find_min_net_sumsub(self, sumsub_static_list, sumratio_static_min_netname, reflist):
        if sumsub_static_list.shape[0] != 0:
            sumratio_minnet_subsum_list = list()
            sumsub_columns = sumsub_static_list.columns
            #sumratio_min_netname = Series(sumratio_static_min_netname, index = sumsub_columns)
            for date_number in sumsub_columns:
                minnet_name = sumratio_static_min_netname.ix[date_number]
                sumratio_minnet_subsum = sumsub_static_list[date_number][minnet_name]
                sumratio_minnet_subsum_list.append(sumratio_minnet_subsum)
            sumratio_minnet_subsum_list = Series(sumratio_minnet_subsum_list, index=sumsub_columns)    
            return sumratio_minnet_subsum_list
        else:
            return reflist


    


    
