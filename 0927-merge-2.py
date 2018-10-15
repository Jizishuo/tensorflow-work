import pandas as pd
import numpy as np
'''
data_15 = pd.read_csv('data/123.csv', encoding='utf-8')
print(data_15)
#a = (data_15.groupby(['year', 'team']).sum().loc[lambda df: df.r > 100])
#print(a)
df = data_15['a'].groupby(data_15['f']).mean()
print(df)
for name, group in data_15.groupby(data_15['w']):
    print(name)
    print(group)
states=np.array([1,1,2,2,2])
years=np.array([1,2,3,4,4])
a = data_15['a'].groupby([states,years]).mean()
print(a)
'''
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
                            'day', 'hour', '小区名称', 'MR-RRC连接建立最大用户数','小区载频PUSCH可用的PRB个数']]

    return result

#取3个字段并处理时间
data_15_1 = Time_chuli(data_15[['开始时间', '小区名称', 'MR-RRC连接建立最大用户数','小区载频PUSCH可用的PRB个数']])
#print(data_15_1)
#df = data_15['a'].groupby(data_15['f']).mean()
#print(df)
#print(data_15_1)
hours=np.array(data_15_1['hour'])
cells=np.array(data_15_1['小区名称'])
print(cells)
a = data_15['MR-RRC连接建立最大用户数'].groupby([hours,cells]).mean()
b = data_15[['MR-RRC连接建立最大用户数','小区载频PUSCH可用的PRB个数']].groupby([cells,hours]).mean()
#print(b)

#b.to_csv('111.csv', encoding='utf_8_sig')