import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
#去掉表头
import chardet
import csv
path = 'data\\82.csv'
path2 = 'data\\82-2.csv'
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
#{'encoding': 'GB2312', 'confidence': 0.99, 'language': 'Chinese'}
'''

zhibiao82 = pd.read_csv('data\\82-2.csv', encoding='utf_8_sig')
#print(zhibiao82)

#cell_1 = zhibiao82[zhibiao82['小区']] == 131
#cell_rrc = zhibiao82['MR-RRC连接建立最大用户数']
dfa = zhibiao82[['MR-RRC连接建立最大用户数', '下行QPSK方式调度TB个数']].loc[zhibiao82['小区'] == 131]
#print(cell_rrc)
print(dfa)
dfa.plot()
plt.show()


