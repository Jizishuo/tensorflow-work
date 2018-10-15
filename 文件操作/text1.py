import csv
import codecs
'''
#w = codecs.open('lte2222.csv', 'w', encoding='utf_8_sig')
#w = open('example.csv', 'w', newline='', encoding='utf_8_sig')
#w1 = csv.writer(w)

with open('lte.CSV', encoding='utf-16') as f:
    reader = csv.reader(f)
    for row in reader:
        print(reader.line_num, row)
        #d = str(row).split('\t')
        #print(d)
        #w1.writerow(row)

w.close()
'''


data = csv.reader(open('lte.CSV','r', encoding='utf-16'), delimiter=',')
print(data)
for i in data:
    print(i)


