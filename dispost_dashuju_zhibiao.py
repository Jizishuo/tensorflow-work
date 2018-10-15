"""
10.2大数据指标出来
"""

import pandas as pd
import numpy as np
'''
am_1002 = pd.read_csv('data/encode/zhibiao/1002_am.csv',encoding='utf_8_sig')
pm_1002 = pd.read_csv('data/encode/zhibiao/1002_pm.csv',encoding='utf_8_sig')
#print(am_1002.head(10))
#	 YY-E-RAB连接建立成功率分母	YY-切换成功率分母	MR-总流量（KB）	MR-RRC连接建立最大用户数_1437649632929
zhibiao_list = ['开始时间', 'eNodeB', 'eNodeB名称', 'YY-RRC连接建立成功率分母',\
                'YY-E-RAB连接建立成功率分母', 'YY-切换成功率分母', 'MR-总流量（KB）',\
                'MR-RRC连接建立最大用户数_1437649632929']
am_1002 = am_1002[zhibiao_list]
print(am_1002.shape)
pm_1002 = pm_1002[zhibiao_list]
print(pm_1002.shape)

result = pd.concat([am_1002, pm_1002], axis=0,  sort=False)
result.to_csv('data/encode/zhibiao/1002_all.csv', encoding='utf_8_sig')

'''

'''
all_1002 = pd.read_csv('data/encode/zhibiao/1002_all.csv',encoding='utf_8_sig')
#print(all_1002.head(10))
#print(all_1002['开始时间'])

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
    result = pd.concat([input_data, time_df], axis=1, sort=False).drop(['开始时间'], axis=1)
    return result

all_1002 = Time_chuli(all_1002)[['year', 'month', 'day', 'hour',  'eNodeB', 'eNodeB名称', 'YY-RRC连接建立成功率分母',\
                'YY-E-RAB连接建立成功率分母', 'YY-切换成功率分母', 'MR-总流量（KB）',\
                'MR-RRC连接建立最大用户数_1437649632929']]

all_1002 = all_1002.sort_values(by=["eNodeB", "hour"]).reset_index(drop=True)
print(all_1002)
#all_1002.to_csv('data/encode/zhibiao/1002_all_2.csv', encoding='utf_8_sig')


cells=np.array(all_1002['eNodeB'])
b = all_1002[['year', 'month', 'day', 'hour', 'YY-RRC连接建立成功率分母',\
                'YY-E-RAB连接建立成功率分母', 'YY-切换成功率分母', 'MR-总流量（KB）',\
                'MR-RRC连接建立最大用户数_1437649632929']].groupby([cells,]).mean()
b.to_csv('data/encode/zhibiao/1002_all_3.csv', encoding='utf_8_sig')
'''


all_1002 = pd.read_csv('data/encode/zhibiao/1002_all_2.csv', encoding='utf_8_sig', index_col=0)
alarm_1002 = pd.read_csv('data/encode/alarm/all_1002_2.csv', encoding='utf_8_sig')
#print(alarm_1002)

#print(all_1002)#1626955--left1629623
#print(alarm_1002)#2966
#'eNodeB', 'year', 'month', 'day', 'hour'
result = pd.merge(all_1002, alarm_1002, on=['eNodeB',], how='left')
print(result)
#result.to_csv('text.csv', encoding='utf_8_sig')


