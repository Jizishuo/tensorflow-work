# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:53:18 2018

@author: User
"""
import pandas as pd
from pandas import DataFrame, Series

zhibiao = pd.read_csv('F:/work/TFTS/data/zhibiao_liuliang_input.csv')
zhibiao.index = zhibiao['cellname']
net_num = list(set(zhibiao['cellname']))
zhibiao2 = zhibiao.drop('cellname', axis=1)
#net_num1 = net_num[:1200]
net1 = DataFrame(zhibiao2.loc['CL314']['flow'])
#net1 = zhibiao2.loc['CL314']['flow']
net1.columns = ['CL314']
net1.index = range(len(net1))
for net in net_num:
    if net != 'CL314':
        net2 = DataFrame(zhibiao2.loc[net]['flow'])
        #net2 = zhibiao2.loc[net]['flow']
        net2.columns = [net]
        net2.index = range(len(net2))
        net1 = pd.concat([net1, net2], axis =1)
         
 
