import pandas as pd
import datetime
from change_GA.GA import GA
import shutil
import os
import time
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import csv
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

'''
    根据名称构建模型
'''
def build_model(name):
    if name.lower() == "lgb":
        model = lgb.LGBMRegressor()
    elif name.lower() == "gdbt":
        model = GradientBoostingRegressor()
    else:
        model = RandomForestRegressor()
    return model


'''
    不重新建模，使用已经构建好的模型
'''
def build_training_model(name):
    if name.lower() == "lgb":
        model = joblib.load('./fcx/files100/lgb/lgb.pkl')
    elif name.lower() == "gbdt":
        model = joblib.load('./fcx/files100/gbdt/gbdt.pkl')
    elif name.lower() == "rf":
        model = joblib.load('./fcx/files100/rf/rf.pkl')
    elif name.lower() == 'xgb':
        model = joblib.load('./fcx/files100/xgb/xgb.pkl')
    elif name.lower() == 'ada':
        model = joblib.load('./fcx/files100/ada/ada.pkl')
    else:
        raise Exception("[!] There is no option ")
    return model


'''
    贝叶斯的黑盒模型，传入参数，计算target（根据模型预测参数的执行时间）
'''
def black_box_function(x):
    print(x)
    print(' ')
    i = []
    model = build_training_model(name)
    # for conf in vital_params['vital_params']:
    #     i.append(params[conf])
        # print(conf)
    # print(i)
    y = model.predict(np.matrix(x))[0]
    # print(y)
    print(y)
    return y





if __name__ == '__main__':
    name='ada'


    # 重要参数
    vital_params_path = './fcx/files100/' + name + "/selected_parameters.txt"
    # 维护的参数-范围表
    conf_range_table = 'Spark_conf_range_wordcount.xlsx'
    # 保存的每一代的最优参数
    generation_confs = "./result/generationBestConf.csv"
    # 保存 GA的参数
    ga_confs_path ="./result/gaConfs.csv"
    # 保存所有的 Y
    all_history_Y_save_path ='./result/all_history_y.csv'




    # 读取重要参数
    vital_params = pd.read_csv(vital_params_path)
    # 参数范围和精度，从参数范围表里面获取
    sparkConfRangeDf = pd.read_excel(conf_range_table)
    sparkConfRangeDf.set_index('SparkConf', inplace=True)
    confDict = sparkConfRangeDf.to_dict('index')
    # 遍历训练数据中的参数，读取其对应的参数空间
    confLb = []  # 参数空间上界
    confUb = []  # 参数空间下界
    precisions = []  # 参数精度
    for conf in vital_params['vital_params']:
        if conf in confDict:
            confLb.append(confDict[conf]['min'])
            confUb.append(confDict[conf]['max'])
            precisions.append(confDict[conf]['pre'])
        else:
            print('-----该参数没有维护: ', conf, '-----')
    # 确定其他参数
    fitFunc = black_box_function  # 适应度函数
    nDim = len(vital_params)  # 参数个数
    sizePop = 40   # 种群数量
    maxIter = 20   # 迭代次数
    probMut = 0.1  # 变异概率
    # 调用遗传算法，记录整个搜索时间
    startTime = datetime.datetime.now()
    ga = GA(func=fitFunc, n_dim=nDim, size_pop=sizePop, max_iter=maxIter, prob_mut=probMut, lb=confLb, ub=confUb,
            precision=precisions)
    best_x, best_y = ga.run()
    endTime = datetime.datetime.now()
    searchDuration = (endTime - startTime).seconds

    # 存储参数配置
    headers = ['func', 'n_dim', 'size_pop', 'max_iter', 'prob_mut', 'lb', 'ub', 'precision', 'searchDuration']
    dicts = [{
        'func': fitFunc, 'n_dim': nDim, 'size_pop': sizePop, 'max_iter': maxIter,
        'prob_mut': probMut, 'lb': confLb, 'ub': confUb, 'precision': precisions,
        'searchDuration': searchDuration
    }]
    with open(ga_confs_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(dicts)
    # 存储每一代的最优解及其结果
    generation_best_X = pd.DataFrame(ga.generation_best_X)
    generation_best_X.columns = vital_params["vital_params"]
    generation_best_X['runtime'] = ga.generation_best_Y
    generation_best_X.to_csv(generation_confs, index=False)
    # 存储所有搜索历史结果
    pd.DataFrame(ga.all_history_Y).to_csv(all_history_Y_save_path)



