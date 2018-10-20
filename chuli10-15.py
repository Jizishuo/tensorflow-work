"""
处理大数据指标和告警--初步处理
"""

import os
import numpy as np
import pandas as pd
import csv
import time


def encode_chuli(path_in, path_out):
    """
    处理文件编码-utf_8_sig
    :param path_in: 输入的路径
    :param path_out: 输出的路径
    :return: none
    """

    f = open(path_in, 'r')
    w = open(path_out, 'w', newline='', encoding='utf_8_sig')

    reader = csv.reader(f)
    writer = csv.writer(w)

    if path_in[-9:] == '(所有列).csv':
        for i, row in enumerate(reader):
            if i >= 1:
                writer.writerow(row)
    else:
        for i, row in enumerate(reader):
            writer.writerow(row)

    w.close()
    f.close()
    #os.remove(path_in)


def chuli_encode(file_day):
    """
    处理文件编码流程
    :return: 保存的路径字典
    """
    path = os.getcwd() + '\data'
    file_list = os.listdir(path)

    zhibiao_list = []
    # 输出的路径字典
    path_dict = {}

    for filename in file_list:
        if filename[-15:] == '历史告警查询(所有列).csv':

            path_in = ('data\\' + filename).replace('\\', '/')
            path_out = ('data\\%s_history.csv' % file_day).replace('\\', '/')
            path_dict['history'] = path_out

            encode_chuli(path_in, path_out)
            print("历史告警编码处理完毕")


        elif filename[-15:] == '当前告警查询(所有列).csv':

            path_in = ('data\\' + filename).replace('\\', '/')
            path_out = ('data\\%s_nowday.csv' % file_day).replace('\\', '/')
            path_dict['nowday'] = path_out

            encode_chuli(path_in, path_out)
            print("当前告警编码处理完毕")


        elif filename != '(所有列).csv':
            zhibiao_list.append(filename)

    if len(zhibiao_list) > 2:
        print("存在多余文件,停止运行")
        exit()
    else:
        for i, filename in enumerate(zhibiao_list):
            path_in = ('data\\' + filename).replace('\\', '/')
            path_out = ('data\\%s_%s.csv' % (file_day, i)).replace('\\', '/')
            path_dict[i] = path_out

            encode_chuli(path_in, path_out)
            if i == 0:
                print("上午文件编码处理完毕")
            else:
                print("下午文件编码处理完毕")
    return path_dict


def Time_chuli(input_data):
    """
    表格时间处理
    :param input_data: #2018-09-26 22:45:00
    :return: {'year': year, 'month': month, 'day':day, 'hour':hour}
    """
    time_list = input_data['发生时间']
    year = list()
    month = list()
    day = list()
    hour = list()
    # minute = list()
    for net in time_list:
        # 2018-09-26 22:45:00
        year.append(int(net[:4]))
        month.append(int(net[5:7]))
        day.append(int(net[8:10]))
        hour.append(int(net[11:13]))
        # minute.append(net[14:16])
    time_df = pd.DataFrame({'year': year, 'month': month, 'day': day, 'hour': hour})
    result = pd.concat([input_data, time_df], axis=1, sort=False).drop(['发生时间'], axis=1)
    return result


def chuli_alarm(path_dict):
    """
    处理基站告警表流程，
    :return: 表（基站，告警， 告警时间）
    """
    history = path_dict.get("history")
    nowdays = path_dict.get("nowday")

    # print(history, nowdays)
    history = pd.read_csv(history, encoding='utf_8_sig', low_memory=False)[['发生时间', '告警恢复时间', '告警级别', '站点ID(局向)']]
    nowday = pd.read_csv(nowdays, encoding='utf_8_sig', low_memory=False)[['发生时间', '告警级别', '站点ID(局向)']]

    def Alary_chuli(items):
        # print(items)
        if items == '严重':
            return 4
        if items == '主要':
            return 3
        if items == '警告':
            return 2
        else:
            return 1

    history['告警级别'] = history['告警级别'].apply(Alary_chuli)
    nowday['告警级别'] = nowday['告警级别'].apply(Alary_chuli)

    # 历史告警 合并重复小区
    enid_h = np.array(history['站点ID(局向)'])
    a1 = history[['发生时间']].groupby([enid_h]).min()
    b1 = history[['告警级别']].groupby([enid_h]).max()
    c1 = history[['告警恢复时间']].groupby([enid_h]).max()

    history_1 = pd.concat([a1, b1, c1], axis=1)
    history_1['eNodeBid'] = history_1.index
    history_1 = history_1.reset_index(drop=True)

    # 当前告警 合并重复小区
    enid_n = np.array(nowday['站点ID(局向)'])
    a2 = nowday[['发生时间']].groupby([enid_n]).min()
    b2 = nowday[['告警级别']].groupby([enid_n]).max()

    nowday_1 = pd.concat([a2, b2], axis=1)
    nowday_1['eNodeBid'] = nowday_1.index
    nowday_1 = nowday_1.reset_index(drop=True)

    # 时间格式处理
    history_1 = Time_chuli(history_1)
    nowday_1 = Time_chuli(nowday_1)

    # 历史告警 告警恢复时间处理
    day = int(list(history_1['day'])[0])

    def Ok_hour(hours):
        # 把超过当天的，没有恢复的 统一为当天最晚23
        if int(hours[8:10]) >= day + 1:
            return 23
        else:
            return hours[11:13]

    history_1['ok_hour'] = history_1['告警恢复时间'].apply(Ok_hour)
    history_1 = history_1.drop(['告警恢复时间'], axis=1)

    # 告警级别  eNodeBid  year month day hour ok_hour---history #1462
    # 告警级别  eNodeBid  year month day hour-----------nowday 324

    # 默认当天告警恢复时间是23点
    # 合并当前和历史表 hour处理成int类型方便计算str-int
    # 重复告警，恢复的故障归类
    nowday_1['ok_hour'] = 23
    history_1['ok_hour'] = history_1['ok_hour'].map(int)

    alary_all = pd.concat([nowday_1, history_1], axis=0).reset_index(drop=True)[
        ['eNodeBid', '告警级别', 'hour', 'ok_hour']]  # 1786

    eNodeBid = np.array(alary_all['eNodeBid'])
    alary_all_1 = alary_all[['告警级别', 'ok_hour']].groupby([eNodeBid]).max().reset_index()  # 1582

    alary_all_2 = alary_all[['hour']].groupby([eNodeBid]).min().reset_index()

    alary_all = pd.merge(alary_all_1, alary_all_2, on='index')

    alary_all.rename(columns={'index': 'eNodeBid'}, inplace=True)
    alary_all['hour'] = alary_all['hour'].map(int)
    alary_all['ok_hour'] = alary_all['ok_hour'].map(int)

    # print(alary_all)  # eNodeBid  告警级别  ok_hour  hour

    def dropsame(hour, ok_hour):
        if hour == ok_hour:
            return np.nan
        else:
            return hour

    # 把发生时间和恢复时间的去除一个(可以不用去除)
    # alary_all['hour'] = alary_all.apply(lambda row: dropsame(row['hour'], row['ok_hour']), axis=1)

    alary_all_F = alary_all[['eNodeBid', '告警级别', 'hour']].dropna()

    alary_all_L = alary_all[['eNodeBid', '告警级别', 'ok_hour']].dropna()
    alary_all_L.rename(columns={'ok_hour': 'hour'}, inplace=True)

    print('告警列表合并完毕')
    return alary_all_F, alary_all_L


def chuli_zhibiao(path_dict):
    """
    处理指标表流程
    :param path_dict: 路径字典
    :return: 指标表（天）
    """

    zhibi_1 = path_dict.get(0)
    zhibi_2 = path_dict.get(1)
    time_list_frist = ['开始时间', 'eNodeB']

    zhibi_1 = pd.read_csv(zhibi_1, encoding='utf_8_sig', low_memory=False)
    zhibiao_list_first_1 = list(zhibi_1.columns[13:])
    all_list_frist = time_list_frist + zhibiao_list_first_1
    zhibi_1 = zhibi_1[all_list_frist]

    zhibi_2 = pd.read_csv(zhibi_2, encoding='utf_8_sig', low_memory=False)[all_list_frist]
    zhibiao = pd.concat([zhibi_1, zhibi_2], axis=0)

    # 设置colums
    zhibiao_name_dict_0 = {'level_0': 'eNodeBid', 'level_1': '发生时间'}
    zhibiao_name_dict_1 = dict()
    for key, value in enumerate(zhibiao_list_first_1):
        zhibiao_name_dict_1[value] = 'zhibiao-%s' % (key + 1)
    zhibiao_name_dict = dict(zhibiao_name_dict_0, **zhibiao_name_dict_1)

    # 合并重复小区，--基站
    enid = np.array(zhibiao['eNodeB'])
    time_id = np.array(zhibiao['开始时间'])
    zhibiao_all = zhibiao[zhibiao_list_first_1].groupby([enid, time_id]).sum().reset_index()

    zhibiao_all.rename(columns=zhibiao_name_dict, inplace=True)
    zhibiao_all = Time_chuli(zhibiao_all)
    zhibiao_all['hour'] = zhibiao_all['hour'].map(int)
    zhibiao_all.to_csv('zhibiao_all.csv', encoding='utf_8_sig', index=False)
    print('合并上下午基站完成')
    return None


def merger_all(alary_all_F, alary_all_L, file_day):
    """
    告警加进指标列表
    :return: none
    """
    zhibiao_data = pd.read_csv('zhibiao_all.csv', encoding='utf_8_sig')  # 302299
    print("读取全部基站")

    result = pd.merge(zhibiao_data, alary_all_F, on=['eNodeBid', 'hour'], how='left')
    result = pd.merge(result, alary_all_L, on=['eNodeBid', 'hour'], how='left')

    eNodeBid = list(set(list(result['eNodeBid'])))  # 12676

    zhibiao_list_first_1 = list(zhibiao_data.columns)

    colmuns_list_all = zhibiao_list_first_1 + ['告警级别_x', '告警级别_y']

    ALL_RESULT = pd.DataFrame(columns=colmuns_list_all)

    def add_enodebid(x):
        ALL_1 = pd.DataFrame(columns=colmuns_list_all)
        for i, cell_id in enumerate(x):
            eNodeBid_list = result.loc[result['eNodeBid'] == cell_id]  # 分小区

            eNodeBid_list['告警级别_x'].ffill()  # .fillna(method='pad')
            eNodeBid_list['告警级别_y'].bfill()  # .fillna(method='bfill')
            ALL_1 = pd.concat([ALL_1, eNodeBid_list], axis=0)
        return ALL_1

    # 分批聚合每批次500个基站添加
    for i in range(len(eNodeBid) // 500 + 1):
        B = eNodeBid[i * 500: (i + 1) * 500]
        AAA = add_enodebid(B)
        ALL_RESULT = pd.concat([ALL_RESULT, AAA], axis=0)

    def all_alary(x, y):
        if x != np.nan and y != np.nan:
            return (x + y) / 2
        else:
            return np.nan

    ALL_RESULT['alary'] = ALL_RESULT.apply(lambda row: all_alary(row['告警级别_x'], row['告警级别_y']), axis=1)
    ALL_RESULT = ALL_RESULT.drop(['告警级别_x', '告警级别_y'], axis=1).fillna(0)

    print('%s的数据共%s行' % (file_day, ALL_RESULT.shape[0]))
    ALL_RESULT.to_csv('%s.csv' % file_day, encoding='utf_8_sig', index=False)
    os.remove('zhibiao_all.csv')

    print("处理完成")
    return None


def add_in_all(file_day):
    data = pd.read_csv('%s.csv' % file_day, encoding='utf_8_sig')
    data_all = pd.read_csv('all.csv', encoding='utf_8_sig')
    ALL_DATA = pd.concat([data_all, data], axis=0)
    print('总共%s行' % ALL_DATA.shape[0])
    ALL_DATA.to_csv('all.csv', encoding='utf_8_sig', index=False)
    print("加进总表-完成")
    return None


def main():
    s = time.time()
    file_day = int(1015)
    path_dict = chuli_encode(file_day)
    alary_all_F, alary_all_L = chuli_alarm(path_dict)
    chuli_zhibiao(path_dict)
    merger_all(alary_all_F, alary_all_L, file_day)

    #add_in_all(file_day)
    print("运行了%s秒" % int((time.time() - s)))


if __name__ == "__main__":
    main()
