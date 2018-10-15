import pandas as pd

print('strat')
g2_1 = pd.read_csv('0924\\2（在线）.xls', '\t', encoding='utf_16')
g2_2 = pd.read_csv('0924\\2（历史）.xls', '\t', encoding='utf_16')
g4_1 = pd.read_csv('0924\\4（在线）.xls', '\t', encoding='utf_16')
g4_2 = pd.read_csv('0924\\4（历史）.xls', '\t', encoding='utf_16')
print('读取完成')
zhibiao_list1 = ['区域', '保障任务', '发生时间', '完成时间', 'CELLID', '基站中文名', '监控内容', '备注', '原因']
zhibiao_list2 = ['区域', '保障内容', '发生时间', '完成时间', 'CELLID', '基站中文名', '监控内容', '备注', '原因']
zhibiao_list3 = ['所属区域', '保障任务', '发生时间', '完成时间', 'CELLID', '监控内容', '备注', '原因', 'ALARM_ID']

# 读取并去掉nan--改2g列名称
g2_1_r = g2_1[zhibiao_list1].fillna({'完成时间': 0}).dropna(axis=0, how='any')
g2_2_r = g2_2[zhibiao_list2].fillna({'完成时间': 0}).dropna(axis=0, how='any')
g2_2_r.rename(columns={'保障内容': '保障任务'}, inplace=True)

g4_1_r = g4_1[zhibiao_list3].fillna({'完成时间': 0}).dropna(axis=0, how='any')
g4_2_r = g4_2[zhibiao_list3].fillna({'完成时间': 0}).dropna(axis=0, how='any')

# 合并2 4g
g2 = pd.concat([g2_1_r, g2_2_r], axis=0, sort=False)
g4 = pd.concat([g4_1_r, g4_2_r], axis=0, sort=False)

# 筛选
g2_data = g2.loc[g2['保障任务'].str.contains('Evip') == True]
g4_data = g4.loc[g4['保障任务'].str.contains('Evip') == True]
print('筛选完成')

#2g处理
g2_df = g2_data['基站中文名'] + '(' + g2_data['CELLID'] + ')'
g2_result = pd.concat([g2_data, g2_df], axis=1)
g2_result.columns = ['区域', '保障任务', '发生时间', '完成时间', 'CELLID', '基站中文名', '监控内容', '备注', '原因', '基站/小区']
g2_result['基站中文名'] = '2g'
g2_result.rename(columns={'基站中文名': '网络类型'}, inplace=True)
g2_result_1 = g2_result[['区域', '保障任务', '发生时间', '完成时间', '基站/小区', '监控内容', '备注', '原因', '网络类型']]

#4g处理
g4_data['ALARM_ID'] = '4g'
g4_data.rename(columns={'所属区域': '区域', 'CELLID': '基站/小区', 'ALARM_ID': '网络类型'}, inplace=True)
print('处理完成')

result_all = pd.concat([g2_result_1, g4_data], axis=0, ignore_index=True)

print(result_all)
result_all.to_csv('evip.csv', encoding='utf_8_sig')
