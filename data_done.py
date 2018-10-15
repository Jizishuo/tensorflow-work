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
import winreg
import time

class data_done(object):
        
    def __get_desktop(self):  
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')  
        return winreg.QueryValueEx(key, "Desktop")[0]  


    def __str2time_1(self, list_in):
        date_list_new = list()
        for time_id in list_in:
            date_new = datetime.datetime.strptime(time_id,'%Y/%m/%d/%H')
            date_new = date_new.strftime('%Y-%m-%d %H:%M:%S.000')
            date_list_new.append(date_new)
        return date_list_new

    def __date_list_1(self, zhibiao):
        d_year = zhibiao['year']
        d_month = zhibiao['month']
        d_day = zhibiao['day']
        d_hour = zhibiao['hour']
        date_len = len(d_year)
        date_haha = range(date_len)
        date_list1 = list(map(lambda i :str(d_year[i])+ '/' + str(d_month[i]) + '/'
                              + str(d_day[i])+ '/' + str(d_hour[i]) , date_haha))
        date_list_out = self.__str2time_1(date_list1)
        return date_list_out

    
    
    def __data_full(self, da_input, date_lost_list_netnum):
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
        zhibiao_out_last_haha = self.__date_caculate(zhibiao_out_haha)   
        return zhibiao_out_last_haha
    
    def __wanzhengxing(self, da_input, cell_name):
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


    def __date_caculate(self, in_data):
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
        zhibiao_input['hour'] = time_list
        zhibiao_input['day'] = day_list
        zhibiao_input['month'] = month_list
        zhibiao_input['year'] = year_list    
        return zhibiao_input



    def __getExePath(self):
        sap = '/'
        if sys.argv[0].find(sap) == -1:
            sap = '\\'
        indx = sys.argv[0].rfind(sap)
        path = sys.argv[0][:indx] + sap
        return path


    def data_out(self):
        path = self.__get_desktop()
        pwd = self.__getExePath()
        file_need_path = pwd + 'basic_data' + os.path.sep
        file_path = path + os.path.sep
        while not(os.path.exists(file_path +'today.csv')):
            print('请在桌面放置today.csv文件')
            time.sleep(5)
        data1 = pd.read_csv(file_path +'today.csv')
        zhibiao_dropNa = data1[['开始时间', '小区名称', 'MR-RRC连接建立最大用户数_1437649632929',
                                'YY-RRC连接建立成功率分母', 'YY-E-RAB连接建立成功率分母',
                                'YY-切换成功率分母', 'MR-总流量（KB）', '所属网格']]
        zhibiao_dropNa.columns = ['time', 'cellname', 'ue', 'rrc', 'erab', 'handover', 'flow', 'net_num']

        zhibiao_dropNa_ca = self.__date_caculate(zhibiao_dropNa)
        full_cellname = pd.read_csv(file_need_path + 'cellname_4loc.csv')

        cell_net = pd.read_csv(file_need_path + 'cell_200net.csv')
        cell_net_need = cell_net[['Description', 'CellName']]
        cell_net_need.columns = ['net_num', 'cellname']

        date_lost_list = self.__wanzhengxing(zhibiao_dropNa_ca, full_cellname)

        date_lost_list_netnum = pd.merge(date_lost_list, cell_net_need, on='cellname', how='inner')


        zhibiao_full_last = self.__data_full(zhibiao_dropNa_ca, date_lost_list_netnum)


        zhibiao_grouped = zhibiao_full_last.groupby(['net_num', 'year', 'month', 'day', 'hour'])

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
       
        #result.to_csv(file_path + 'result_today.csv')
    
        return result
    

