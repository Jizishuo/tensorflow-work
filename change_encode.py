import chardet
import datetime
import time
import csv
path = 'data/1001_pm.csv'
path2 = 'data/1001_pm_1.csv'

#startime =datetime.datetime.now()
startime =time.time()
f = open(path, 'r')
w = open(path2, 'w', newline='', encoding='utf_8_sig')

reader = csv.reader(f)
writer = csv.writer(w)
for i,row in enumerate(reader):
    #if i > 4:
        #print(i, row)
    writer.writerow(row)

w.close()
f.close()

#endtime = datetime.datetime.now()
endtime = time.time()
print(startime-endtime)#86387 -12.174696445465088
#print(chardet.detect(data))
#{'encoding': 'GB2312', 'confidence': 0.99, 'language': 'Chinese'}