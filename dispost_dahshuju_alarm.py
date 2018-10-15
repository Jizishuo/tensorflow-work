"""
10，2大数据告警处理
"""


import pandas as pd
import numpy as np
'''
data_1001_nowdays = pd.read_csv('data/encode/alarm/1002_nowdays.csv', encoding='utf_8_sig')
#time_di = data_1001_history[['发生时间', '告警恢复时间', '告警码', '告警级别', '站点名称(局向)', '站点ID(局向)']]
time_di = data_1001_nowdays[['发生时间', '告警码', '告警级别', '站点名称(局向)', '站点ID(局向)']]
print(time_di)
time_di.to_csv('data/encode/alarm/1002_nowdays_1.csv',encoding='utf_8_sig')
'''


nowdays_1002 = pd.read_csv('data/encode/alarm/1002_nowdays_1.csv',encoding='utf_8_sig').dropna()
nowdays_1002 = nowdays_1002[['发生时间', '告警码', '告警级别', '站点ID(局向)']]

def Alary_chuli(items):
    #print(items)
    if items =='严重':
        return 4
    if items =='主要':
        return 3
    if items =='警告':
        return 2
    else:
        return 1

nowdays_1002['告警级别'] = nowdays_1002['告警级别'].apply(Alary_chuli)
#print(nowdays_1002)

enid=np.array(nowdays_1002['站点ID(局向)'])
a1 = nowdays_1002[['发生时间']].groupby([enid]).min()
b1 = nowdays_1002[['告警级别']].groupby([enid]).mean()

nowdays_1002_1 = pd.concat([a1, b1], axis=1)
nowdays_1002_1['eNodeB'] = nowdays_1002_1.index
nowdays_1002_1 = nowdays_1002_1.reset_index(drop=True)
#print(nowdays_1002_1)

def Time_chuli(input_data):
    time_list = input_data['发生时间']
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
    result = pd.concat([input_data, time_df], axis=1, sort=False).drop(['发生时间'], axis=1)
    return result
nowdays_1002 = Time_chuli(nowdays_1002_1) #310
nowdays_1002['ok_hour'] = 23

#print(nowdays_1002.shape)


history_1002 = pd.read_csv('data/encode/alarm/1002_history_1.csv',encoding='utf_8_sig').dropna()
history_1002 = history_1002[['发生时间', '告警恢复时间', '告警级别', '站点ID(局向)']]


history_1002['告警级别'] = history_1002['告警级别'].apply(Alary_chuli)

enid=np.array(history_1002['站点ID(局向)'])
a2 = history_1002[['发生时间']].groupby([enid]).min()
b2 = history_1002[['告警级别']].groupby([enid]).mean()
c2 = history_1002[['告警恢复时间']].groupby([enid]).max()

history_1002_1 = pd.concat([a2, b2, c2], axis=1)
history_1002_1['eNodeB'] = history_1002_1.index
history_1002_1 = history_1002_1.reset_index(drop=True)

history_1002 = Time_chuli(history_1002_1)


def Ok_hour(hours):
    #print(hours)
    if int(hours[8:10]) >= 3:
        return 23
    else:
        return hours[11:13]

history_1002['ok_hour'] = history_1002['告警恢复时间'].apply(Ok_hour)
history_1002 = history_1002.drop(['告警恢复时间'], axis=1)
#print(history_1002.shape)

result = pd.concat([nowdays_1002, history_1002], axis=0, sort=False)
print(result)
print(len(list(result['eNodeB'])))#1483
print(len(set(list(result['eNodeB']))))#1349
#result.to_csv('data/encode/alarm/all_1002.csv', encoding='utf_8_sig')

#告警级别	enid	year	month	day	hour	ok_hour

result_1 = result.loc[:, ('告警级别','eNodeB', 'year', 'month', 'day', 'hour')]

result_2 = result.loc[:, ('告警级别','eNodeB', 'year', 'month', 'day', 'ok_hour')]
result_2.rename(columns={'ok_hour': 'hour'}, inplace=True)
#print(result_2)1483
result3 = pd.concat([result_1, result_2], axis=0, sort=False)
#result3.to_csv('data/encode/alarm/all_1002_2.csv', encoding='utf_8_sig')
print(result3.shape)#2966

