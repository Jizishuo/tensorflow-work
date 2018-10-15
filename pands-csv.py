import pandas as pd
import datetime
from pandas import DataFrame

original_csv = pd.read_csv("F:\\test-data-ex\\text-5.csv")

titile_list = ['开始时间', '网元', 'eNodeB', 'YY-RRC连接建立成功率分母', 'YY-E-RAB连接建立成功率分母', 'YY-切换成功率分母', 'MR-总流量（KB）', 'MR-RRC连接建立最大用户数_1437649632929']
'''
for i in original_csv.head(1):
    #print(i)
    titile_list.append(i)
print(titile_list)
'''

def time_with(data_in):
    time_out = data_in
    time_co = time_out['time']
    time_list = list()
    day_list = list()
    month_list = list()
    year_list = list()
    for time in time_co:
        time_list.append(time[11:13])
        day_list.append(time[8:10])
        month_list.append(time[5:7])
        year_list.append(time[0:4])
    time_out['time'] = pd.DataFrame(time_list)
    time_out['day'] = pd.DataFrame(day_list)
    time_out['month'] = pd.DataFrame(month_list)
    time_out['year'] = pd.DataFrame(year_list)
    name_list = ['year','month','day','time', 'cellname', 'eNodeB', \
                 'yy-rrc', 'yy-e-eab', 'yy', 'flow', 'rrc']
    return time_out.reindex(columns=name_list)

zhibiao_dropNa = original_csv[titile_list]
zhibiao_dropNa.columns = ['time', 'cellname', 'eNodeB', 'yy-rrc', 'yy-e-eab', 'yy', 'flow', 'rrc']
data = time_with(zhibiao_dropNa)
data.sort_values(["eNodeB"])
data.to_csv("F:\\test-data-ex\\text-out-5.csv")
eNodeB_list = []
eNodeb_all = data["eNodeB"]
for eNodeB in eNodeb_all:
    if eNodeB !="nan":
        eNodeB_list.append(eNodeB)

ids = list(set(eNodeB_list))
print(ids)
print(len(ids))