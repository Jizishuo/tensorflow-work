"""
处理全文cellid
"""
import pandas as pd
import numpy as np


let_data = pd.read_csv('lte.CSV', '\t', encoding='utf_16')[[ '小区标识','ENODEB','小区中文名','NODEBID','小区ID','pool','CGI','VLAN']]
let_data['pool'] = let_data['小区ID'] + let_data['NODEBID']*256
#let_data['VLAN'] = let_data['CGI']
print(let_data)
let_data.to_csv('lte-chuli.csv', encoding='utf_8_sig')