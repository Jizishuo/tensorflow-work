import xlrd

import pandas as pd

#data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\lte2222.csv', encoding='utf-8')
#print(data)


import codecs
import chardet

f = codecs.open('lte.CSV', 'r', encoding='utf-16')
w = codecs.open('lte2222.csv', 'w', encoding='utf_8_sig')
data = f.read()#.decode('utf-16')
print(data)

w.write(data)

w.close()

f.close()

'''
import sys
#reload(sys)
#print (sys.getdefaultencoding()) #ascii
#sys.setdefaultencoding('utf-8')

import base64
fin = open("C:\\Users\\Administrator\\Desktop\\lte.CSV","rb")

fout = open("C:\\Users\\Administrator\\Desktop\\lte333.csv", "w", encoding='utf-8')
base64.encode(fin, fout)
fin.close()
fout.close()

'''


