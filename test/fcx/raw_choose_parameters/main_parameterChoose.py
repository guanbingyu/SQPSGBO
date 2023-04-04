# 读取最初的总数据(包含micro,os,container)，对三层分别进行特征选择

import parameter_choose  # 用于特征选择的类

import pandas as pd

import os


class Main:
    def __init__(self, file_path, save_path,test_data, name, model_name, step, left_num, quantile, target):
        self.global_min_error = 3
        self.global_min_error_location = " "  # 所有模型中误差最低的模型和数据量所在位置
        self.global_min_error_location_eq = " "
        # ["lgb", "rf", "ada", "gdbt", "xgb"]
        self.flag = True  # true 表示训练第一阶段模型， false表示训练第二阶段模型
        self.filepath = file_path
        self.save_path = save_path
        self.name = name
        self.test_data=test_data
        self.target = target  # 训练目标
        self.quantile = quantile  # 执行时间超过该分位数的数据将被去掉
        self.origin_data, self.features_list = self.get_data(file_path=self.filepath, name="parameters")  # 原始训练数据
        self.data_length = len(self.origin_data)
        print(self.data_length)
        # self.interval = interval  # 下一轮训练增加的训练量

        # 以下参数用于迭代训练的类  parameter_choose.Choose
        self.step = step  # 迭代训练，每次去除step个参数
        self.left_num = left_num  # 只剩下left_num个参数时停止训练
        self.model_name = model_name  # 构建模型的名字

    def write_vital_parameters(self, parameters, name, save_path):  # 写入重要配置参数
        vital_params = []
        for i in range(len(parameters)):
            vital_params.append(parameters[i])
        file = open(save_path + name + "selected_parameters.txt", mode="a+")
        file.write("vital_params")
        file.write("\n")
        for feature in vital_params:
            file.write(feature)
            file.write("\n")
        file.close()

    def main(self):
        cur_min_error_location = " "  # 当前模型误差最低点在哪一个数据量中
        cur_min_error = 3.0  # 当前模型误差最低点

        folder=os.path.exists(self.save_path)
        if not folder:
            os.makedirs(self.save_path)
            print('----创建新文件夹----')
        new_save_path = self.save_path + self.model_name + "\\"  # 每种算法建一个文件夹
        print(new_save_path)
        folder = os.path.exists(new_save_path)
        if not folder:
            os.mkdir(new_save_path)
        new_save_path_reversed = new_save_path + "reserved\\"
        folder = os.path.exists(new_save_path_reversed)
        if not folder:
            os.mkdir(new_save_path_reversed)

        # 总数据
        data, features_list = self.get_data(file_path=self.filepath, name="parameters")
        '''
        2022.1.7
        采用真实数据进行测试
        '''
        test_data,features_list=self.get_data(file_path=self.test_data, name="parameters")
        total_choose = parameter_choose.Choose(name=self.model_name, features=features_list, step=self.step,
                                               prefix="parameters",
                                               data=data,
                                               test_data=test_data,
                                               save_path=new_save_path, target=self.target,
                                               left_num=self.left_num)

        total_choose.main()

        if self.flag == True:
            self.write_vital_parameters(parameters=total_choose.final_features, save_path=new_save_path,
                                        name="")
            self.write_vital_parameters(parameters=total_choose.final_features_with_reserved,
                                        save_path=new_save_path_reversed, name="")

    def get_data(self, file_path, name):  # 读取训练数据
        data = pd.read_csv(file_path)

        all_columns = data.columns

        column_length = len(all_columns)

        print("\n")
        print(name + " 特征个数: ")
        print(str(column_length - 1))

        print(name + " 行数: ")
        print(str(len(data)))
        if self.quantile < 1.0:
            time_threshold = data[self.target].quantile(self.quantile)
            new_data = data[data[self.target] < time_threshold]
        else:
            new_data = data
        print(name + " 新数据行数: ")
        print(str(len(new_data)))
        # 存放特征
        features_list = []
        for feature in all_columns:
            if feature != self.target:
                features_list.append(feature)
        return new_data, features_list
