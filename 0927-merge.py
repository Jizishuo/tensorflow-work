"""
pandas合并15分钟粒度处理
"""


'''
import csv
path = 'data\\0927.csv'
path2 = 'data\\0927-1.csv'
f = open(path, 'r')
w = open(path2, 'w', newline='', encoding='utf_8_sig')

reader = csv.reader(f)
writer = csv.writer(w)
for i,row in enumerate(reader):
    if i > 4:
        #print(i, row)
        writer.writerow(row)

#print(chardet.detect(data))
w.close()
f.close()

'''

import pandas as pd
import numpy as np

data_15 = pd.read_csv('data/0927-1.csv', encoding='utf_8_sig')


def Time_chuli(input_data):
    time_list = input_data['开始时间']
    year = list()
    month = list()
    day = list()
    hour =list()
    minute = list()
    for net in time_list:
        #2018-09-26 22:45:00
        year.append(net[:4])
        month.append(net[5:7])
        day.append(net[8:10])
        hour.append(net[11:13])
        #minute.append(net[14:16])
    time_df = pd.DataFrame({'year': year, 'month': month, 'day':day, 'hour':hour})
    result = pd.concat([input_data, time_df], axis=1, sort=False)[['year', 'month', \
                            'day', 'hour', '小区名称', 'MR-RRC连接建立最大用户数']]

    return result

#取3个字段并处理时间
data_15_1 = Time_chuli(data_15[['开始时间', '小区名称', 'MR-RRC连接建立最大用户数']])
#print(data_15_1)
#data_15_1.to_csv('1111.csv', encoding='utf_8_sig')

#建一个新的表
new_data = pd.DataFrame(index=[], columns=['year', 'month', \
                            'day', 'hour', 'cell_name', 'rrc'])

#已小区名称-分表
cell_data = list(set(list(data_15_1['小区名称'])))
#print(cell_data)
for cell_id in cell_data:
    cell_data_list = data_15_1.loc[data_15_1['小区名称']==cell_id]          #分小区
    for hour in sorted(list(set(cell_data_list['hour']))):                  #分时间-去重排序
        cell_time_list = cell_data_list.loc[cell_data_list['hour'] == hour]
        try:
            rrc = [np.array(cell_time_list['MR-RRC连接建立最大用户数']).mean(),] #取rrc平均
            year = list(set(cell_time_list['year']))
            month = list(set(cell_time_list['month']))
            day = list(set(cell_time_list['day']))
            time = list(set(cell_time_list['hour']))
            cell = list(set(cell_time_list['小区名称']))
            merge_df = pd.DataFrame({'year': year, 'month': month, 'day': day, 'hour': time, 'cell_name':cell, 'rrc':rrc})
            new_data = pd.concat([new_data, merge_df], axis=0, sort=False, ignore_index=True)
        except:
            print("基站小区名称日期不对称")

print(new_data)
new_data.to_csv('data/0927-2-merge.csv', encoding='utf_8_sig')