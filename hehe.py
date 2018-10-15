# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 23:57:44 2018

@author: User
"""

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
        zhibiao_input['hour'] = time_list
        zhibiao_input['day'] = day_list
        zhibiao_input['month'] = month_list
        zhibiao_input['year'] = year_list    
        return zhibiao_input
    
    
    da_input = zhibiao_dropNa_ca
    cell_name = full_cellname
    
    def wanzhengxing(da_input, cell_name):
        date_haha = da_input['time']
        dates_list = list(set(list(date_haha)))
        date_day = list()
        for date_x in dates_list:
            date_day.append(date_x[0:11])
        days = len(list(set(date_day)))
        #days = len(dates_list)/24
        date_numbs = 24*days
        cell_list = list(cell_name['cellname'])
        cell_list_input = list(da_input['cellname'])
        date_lost_out = DataFrame(columns=['cellname', 'lost_times'])
        #cell_item = '广州东晖花园F-ZLH-1'
        for cell_item in cell_list:
            net_num_counter = cell_list_input.count(cell_item)
            if net_num_counter < date_numbs:
                print(net_num_counter, date_numbs)
                ccc = DataFrame([cell_item,date_numbs-net_num_counter])
                ddd = ccc.T
                ddd.columns = ['cellname', 'lost_times']
                date_lost_out = date_lost_out.append(ddd)
        return date_lost_out


da_input = zhibiao_dropNa_ca
date_lost_list_netnum = date_lost_list_netnum

def time_stamp_to_list(time_stamp):
        timelist = []
        for stamp in time_stamp:
            timelist.append(stamp.strftime('%Y-%m-%d %H:%M:%S.000'))
        return timelist


def date_full(date_list_in):
    date_day = list()
    for date_x in date_list_in:
        date_day.append(date_x[0:11])
    date_day_set = list(set(date_day))
    day_number = len(date_day_set)
    if day_number == 1:
        hours = ['00', '01', '02', '03','04', '05', '06', '07', '08', '09', '10', 
                 '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                 '21', '22', '23']
        time_list = list()
        for hour_item in hours:
            a = date_day_set[0]
            b = a + "%2s:00:00.000"%hour_item
            time_list.append(b)
        return time_list
    else:
        date_day_set.sort()
        day_start = date_day_set[0]
        day_end = date_day_set[len(date_day_set)-1]
        day_end = day_end.replace(' ','')
        day_end_last = datetime.datetime.strptime(day_end, '%Y-%m-%d')+datetime.timedelta(days=1)         
        time_list = list(pd.date_range(day_start, day_end_last, freq='H'))
        time_list_out = time_stamp_to_list(time_list)[0:len(time_list)-1]
        return time_list
        
    
def data_full(da_input, date_lost_list_netnum):
        zhibiao_out = da_input
        date_list = list(set(list(zhibiao_out['time'])))
        date_list_last = date_full(date_list)
        lost_list = date_lost_list_netnum['cellname']
        lost_net_num = date_lost_list_netnum['net_num']
        dict_lost = dict(zip(lost_list, lost_net_num))
        zhibiao_out_last = DataFrame(columns=zhibiao_out.columns)
    #da_input['time'] = date_list_need
        for item in lost_list:
            item_date = list(zhibiao_out[zhibiao_out.cellname == item]['time'])
            #item_date = list(zhibiao_out[zhibiao_out.cellname == item]['time'])
            item_date_lost = list(set(date_list_last).difference(set(item_date)))
            item_full = DataFrame(columns=zhibiao_out.columns)
            item_full['time'] = item_date_lost
            item_full['cellname'] = item
            item_full['net_num'] = dict_lost[item]
            item_full = item_full.fillna(0)
            zhibiao_out_last = zhibiao_out_last.append(item_full)
        zhibiao_out_haha = zhibiao_out.append(zhibiao_out_last)
        zhibiao_out_last_haha = date_caculate(zhibiao_out_haha)   
        return zhibiao_out_last_haha
    
    def get_desktop():  
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders')  
        return winreg.QueryValueEx(key, "Desktop")[0]  
    
    def getExePath():
        sap = '/'
        if sys.argv[0].find(sap) == -1:
            sap = '\\'
        indx = sys.argv[0].rfind(sap)
        path = sys.argv[0][:indx] + sap
        return path