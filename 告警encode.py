import chardet
import datetime
import time
import csv
path_in = 'F:/python项目/work/zhibiao_dashuju_10_9/data1007-08/1007/(1007)历史告警查询(所有列).csv'
#F:/python项目/work/zhibiao_dashuju_10_9/data1007-08/1007/(1007)历史告警查询(所有列).csv
path_out = 'F:/python项目/work/zhibiao_dashuju_10_9/data_chuli/gaojing/1007_history.csv'
#F:/python项目/work/zhibiao_dashuju_10_9/data_chuli/gaojing/1007_history.csv
path = 'data/(1002)历史告警查询(所有列).csv'
path2 = 'data/1002_history.csv'

#startime =datetime.datetime.now()
startime =time.time()
f = open(path_in, 'r')
w = open(path_out, 'w', newline='', encoding='utf_8_sig')

reader = csv.reader(f)
writer = csv.writer(w)
for i,row in enumerate(reader):
    if i >= 1:
        #print(i, row)
        writer.writerow(row)

w.close()
f.close()

#endtime = datetime.datetime.now()
endtime = time.time()
print(startime-endtime)#86387 -12.174696445465088
#print(chardet.detect(data))
#{'encoding': 'GB2312', 'confidence': 0.99, 'language': 'Chinese'}