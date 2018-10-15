import os
import pandas as pd
import numpy as np

#let_data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\lte.CSV', '\t', encoding='utf_16')
let_data = pd.read_csv('lte2.csv')
df = pd.read_excel('data.xlsx').fillna(0)

def date_caculate(in_data):
    cell_id_date = np.array(in_data['小区ID'])
    nodeb_id_date = np.array(in_data['NODEBID'])
    eic = list(nodeb_id_date*256 + cell_id_date)
    eic_df = pd.DataFrame({'eic': eic})
    frames = pd.concat([in_data, eic_df], axis=1, sort=False)
    return frames

zhibiao_dropNa = let_data[['小区ID', '小区标识', '小区中文名',\
                           '基站名称', 'ENODEB', '优化区域', 'pool', 'NODEBID', 'CGI']]

result = date_caculate(zhibiao_dropNa)
#result.to_csv('first-result2.csv', index=0, encoding='utf_8_sig')

data_in =df[['eic','NODEBID', 'ENODEB','小区标识','CGI']]

eic_data = list(set(list(data_in['eic'])))
NODEBID_data = list(set(list(data_in['NODEBID'])))
ENODEB_data = list(set(data_in['ENODEB']))
cell_data = list(set(list(data_in['小区标识'])))
cgi_data = list(set(list(data_in['CGI'])))

#no_list = result.loc[result['NODEBID'].isin(l2), ["小区标识"]]
eic_data_list = result.loc[result['eic'].isin(eic_data)]
NODEBID_data_list = result.loc[result['NODEBID'].isin(NODEBID_data)]
ENODEB_data_list = result.loc[result['ENODEB'].isin(ENODEB_data)]
cell_data_list = result.loc[result['小区标识'].isin(cell_data)]
cgi_data_list = result.loc[result['CGI'].isin(cgi_data)]

all_list= [eic_data_list, NODEBID_data_list, ENODEB_data_list, cell_data_list, cgi_data_list]
lalal = pd.concat(all_list, keys=['eic','NODEBID','ENODEB','小区标识',	'CGI'])

lalal.to_csv('result.csv',encoding='utf_8_sig')
