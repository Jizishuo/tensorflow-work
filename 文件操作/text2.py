import pandas as pd

data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\lte.CSV', '\t', encoding='utf_16')
#data = pd.read_excel('C:\\Users\\Administrator\\Desktop\\lte.xls', encoding='utf_16')
data.to_csv('result.csv',encoding='utf_8_sig')
