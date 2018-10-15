import os
import time
import csv
import pandas as pd
import numpy as np


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


def chuli_encode():
    """
    处理文件编码流程
    :return: 保存的路径字典
    """
    path = os.getcwd() + '\data1007-08' + '\\1007'
    # path_chuli = os.getcwd() + '\data_chuli'
    path_chuli = 'data_chuli'
    filelist = os.listdir(path)

    zhibiao_list = []
    # 输出的路径字典
    path_dict = {}

    for filename in filelist:
        if filename[-15:] == '历史告警查询(所有列).csv':

            path_in = (path + '\\' + filename).replace('\\', '/')
            path_out = (path_chuli + '\\gaojing\\1007_history.csv').replace('\\', '/')
            path_dict['history'] = path_out

            # encode_chuli(path_in, path_out)
            print("历史告警处理完毕")

        elif filename[-15:] == '当前告警查询(所有列).csv':

            path_in = (path + '\\' + filename).replace('\\', '/')
            path_out = (path_chuli + '\\gaojing\\1007_nowday.csv').replace('\\', '/')
            path_dict['nowday'] = path_out

            # encode_chuli(path_in, path_out)
            print("当前告警处理完毕")

        elif filename != '(所有列).csv':
            zhibiao_list.append(filename)

    if len(zhibiao_list) > 2:
        print("存在多余文件")
    else:
        for i, filename in enumerate(zhibiao_list):
            path_in = (path + '\\' + filename).replace('\\', '/')
            path_out = (path_chuli + '\\zhibiao\\1007_%s.csv' % i).replace('\\', '/')
            path_dict[i] = path_out

            # encode_chuli(path_in, path_out)
            print("第%s文件处理完毕" % (i + 1))
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
    history = pd.read_csv(history, encoding='utf_8_sig')[['发生时间', '告警恢复时间', '告警级别', '站点ID(局向)']]
    nowday = pd.read_csv(nowdays, encoding='utf_8_sig')[['发生时间', '告警级别', '站点ID(局向)']]

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
    alary_all['hour'] = alary_all.apply(lambda row: dropsame(row['hour'], row['ok_hour']), axis=1)

    alary_all_F = alary_all[['eNodeBid', '告警级别', 'hour']].dropna()

    alary_all_L = alary_all[['eNodeBid', '告警级别', 'ok_hour']].dropna()
    alary_all_L.rename(columns={'ok_hour': 'hour'}, inplace=True)

    return alary_all_F, alary_all_L


def chuli_zhibiao(path_dict):
    """
    处理指标表流程
    :param path_dict: 路径字典
    :return: 指标表（天）
    """
    '''
    zhibi_1 = path_dict.get(0)
    zhibi_2 = path_dict.get(1)
    print(zhibi_1, zhibi_2)
    zhibi_1 = pd.read_csv(zhibi_1, encoding='utf_8_sig')[['开始时间', 'eNodeB', 'MR-RRC连接建立最大用户数']]
    zhibi_2 = pd.read_csv(zhibi_2, encoding='utf_8_sig')[['开始时间', 'eNodeB', 'MR-RRC连接建立最大用户数']]
    print(zhibi_1)#810221
    print(zhibi_2)#812917

    zhibiao = pd.concat([zhibi_1, zhibi_2], axis=0)
    print(zhibiao)#1623138
    zhibiao.to_csv('zhibiao.csv', encoding='utf_8_sig')
    '''

    zhibiao = pd.read_csv('zhibiao.csv', encoding='utf_8_sig')  # index_col=0
    # print(zhibiao)#1623138 开始时间  eNodeB  MR-RRC连接建立最大用户数
    # print(len(set(list(zhibiao['eNodeB']))))#12700

    # 合并重复小区，--基站
    enid = np.array(zhibiao['eNodeB'])
    time_id = np.array(zhibiao['开始时间'])
    zhibiao_all = zhibiao[['MR-RRC连接建立最大用户数']].groupby([enid, time_id]).sum().reset_index()

    zhibiao_all.rename(columns={'level_0': 'eNodeBid', 'level_1': '发生时间', 'MR-RRC连接建立最大用户数': 'rrc'}, inplace=True)
    zhibiao_all = Time_chuli(zhibiao_all)
    print(zhibiao_all)
    zhibiao_all['hour'] = zhibiao_all['hour'].map(int)
    # zhibiao_all.to_csv('text.csv', encoding='utf_8_sig', index=False) #302299
    return None


def merger_all(alary_all_F, alary_all_L, zhibiao_data):
    """
    告警加进指标列表
    :return: none
    """
    zhibiao_data = pd.read_csv('text.csv', encoding='utf_8_sig')  # 302299

    result = pd.merge(zhibiao_data, alary_all_F, on=['eNodeBid', 'hour'], how='left')
    result = pd.merge(result, alary_all_L, on=['eNodeBid', 'hour'], how='left')

    eNodeBid = list(set(list(result['eNodeBid'])))  # 12676

    ALL_1 = pd.DataFrame(columns=['eNodeBid', 'rrc', 'year', 'month', 'day', 'hour', '告警级别_x', '告警级别_y'])
    for i, cell_id in enumerate(eNodeBid[:500]):
        eNodeBid_list = result.loc[result['eNodeBid'] == cell_id]  # 分小区

        eNodeBid_list['告警级别_x'] = eNodeBid_list['告警级别_x'].fillna(method='pad')
        eNodeBid_list['告警级别_y'] = eNodeBid_list['告警级别_y'].fillna(method='bfill')
        # result['告警级别_y'] = eNodeBid_list['告警级别_y'].fillna(method='bfill')
        ALL_1 = pd.concat([ALL_1, eNodeBid_list], axis=0)

    def all_alary(x, y):
        if x != np.nan and y != np.nan:
            return (x + y) / 2
        else:
            return np.nan

    ALL_1['alary'] = ALL_1.apply(lambda row: all_alary(row['告警级别_x'], row['告警级别_y']), axis=1)
    ALL_1 = ALL_1.drop(['告警级别_x', '告警级别_y'], axis=1).fillna(0)
    print(ALL_1)

    ALL_1.to_csv('111.csv', encoding='utf_8_sig', index=False)
    print("处理完成")


if __name__ == '__main__':
    # 处理文件编码流程
    path_dict = chuli_encode()

    # 处理告警列表
    alary_all_F, alary_all_L = chuli_alarm(path_dict)

    # 处理指标表
    zhibiao_data = chuli_zhibiao(path_dict)

    # 指标加告警合并
    merger_all(alary_all_F, alary_all_L, zhibiao_data)
