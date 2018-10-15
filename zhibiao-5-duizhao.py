"""
对应网格处理
"""
'''
import csv
path = '0927-5.csv'
path2 = '0927-5-1.csv'
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

wangge_data = pd.read_csv('wangge-200.csv')[['CellID', 'Description']]
zhibiao_5 = pd.read_csv('0927-5-1.csv', encoding='utf_8_sig')[['小区', 'eNodeB', '小区名称', 'eNodeB名称', \
                                                               'YY-RRC连接建立成功率分母', 'YY-E-RAB连接建立成功率分母', \
                                                               'YY-切换成功率分母', 'MR-总流量（KB）', \
                                                               'MR-RRC连接建立最大用户数_1437649632929']]

zhibiao_5 = zhibiao_5.dropna(axis=0, how='any')
zhibiao_5['CellID'] = (zhibiao_5['eNodeB'].map(int).map(str) + zhibiao_5['小区'].map(str)).map(int)

# zhibiao_5.to_csv('zhibiao_5.csv', encoding='utf_8_sig')

hebing_data = pd.merge(zhibiao_5, wangge_data, on='CellID')

hebing_data.columns = [['cell', 'enodeb_id', 'cell_name', 'enodeb_name', \
                        'yy_rrc', 'yy-e-rab', 'yy', 'traffic', 'rrc', 'CellID', 'Description']]


# hebing_data.to_csv('hebing.csv', encoding='utf_8_sig')

Description = np.array(hebing_data['Description']).T   #二维转一维

for i in Description:
    b = hebing_data[['yy_rrc','yy-e-rab','yy','traffic','rrc']].groupby(i).sum()
print(b)
b.to_csv('hebing.csv', encoding='utf_8_sig')
