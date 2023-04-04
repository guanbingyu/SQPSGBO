# coding=UTF-8
# 2021/7/11   对画图加了一些标注
# 2021/7/13  改为用于筛选配置参数
# 2021/7/15更新，训练三次，取重要性和误差的平均值
# 2021/12/3更新，强制保留parametersList里面的重要参数
import datetime
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import parametersList
# 读文件

import time


class Choose:
    def __init__(self, name, features, data,test_data, step, prefix, save_path, target, left_num):
        self.name = name
        self.final_features = []  # 最终选出的特征
        self.final_importance = []  # 最终选出特征的重要性
        self.min_error = 1.0  # 最小误差，最终选择的特征就是误差最低点的特征
        self.error_list = []  # 把每一次迭代的平均误差记录下来
        self.model = lgb.LGBMRegressor()
        self.data = data
        self.test_data=test_data
        self.feature_length = len(features)
        self.step = step  # 每次扔几个特征
        self.init_features = []
        self.init_features = features

        self.save_path = str(save_path)  # 生成文件存储路径

        self.prefix = prefix  # 表明是os,micro,或者container

        self.init_features = set(self.init_features).intersection(data.columns)

        self.target = target
        self.flag = False  # 训练终止标志，True表示结束训练
        self.reserved_parameters = parametersList.parameters_list
        self.left_num = left_num
        self.final_features_with_reserved = []  # 包含保留参数的参数列表
        self.reserved_min_error= 1.0

        print("训练数据行数" + str(len(self.data)))

        # print(len(self.data))

    # 根据名称构建模型
    def build_model(self):
        if self.name.lower() == "lgb":
            model = lgb.LGBMRegressor()
        elif self.name.lower() == "gbdt":
            model = GradientBoostingRegressor()
        elif self.name.lower() == "ada":
            model = AdaBoostRegressor()
        elif self.name.lower() == "xgb":
            model = xgb.XGBRegressor()
        else:
            model = RandomForestRegressor()
        return model

    # 切分数据做标准化
    def process_data(self, features, index):
        # 切分数据集,测试集占0.25

        features_data = self.data[features]
        target_data = self.data[self.target]
        '''
        2022.1.7对测试数据进行更改
        
        '''
        x_train=self.data[features]
        y_train=self.data[self.target]

        x_test=self.test_data[features]
        y_test=self.test_data[self.target]

        # x_train, x_test, y_train, y_test = train_test_split(features_data, target_data, test_size=0.20, random_state=22)

        # 做标准化`
        # transfer = StandardScaler()
        # x_train = transfer.fit_transform(x_train)
        # x_test = transfer.transform(x_test)
        return x_train, x_test, y_train, y_test

    # 计算误差
    def error_calculate(self, y_predict, y_test):
        y_test = y_test.tolist()
        test_length = len(y_test)
        error_percentage = 0
        # print(test_length)

        for i in range(0, test_length):
            # print(y_predict[i])
            # print((abs(y_test[i] - y_predict[i]) / y_predict[i]))
            error_percentage = error_percentage + (abs(y_test[i] - y_predict[i]) / y_test[i])

        # 所有误差取平均值

        error_percentage = error_percentage / test_length

        return error_percentage

    def main(self):
        startTime = datetime.datetime.now()
        print("开始训练")
        print(startTime)
        print("\n")
        self.choose_features(self.init_features)
        endTime = datetime.datetime.now()
        duration = (endTime - startTime).seconds
        minutes_ = duration // 60
        seconds_ = duration - minutes_ * 60
        print("结束训练")
        print(endTime)
        print("\n")
        str_time = str(minutes_) + "m" + str(seconds_) + "s"
        print("训练时间" + str_time)
        print("\n")

        self.plot_error()
        self.plot_feature_importance()
        self.error_to_file(str_time)
        self.features_to_file(str_time)

    def train(self, features, index):
        # 取训练集，测试集
        x_train, x_test, y_train, y_test = self.process_data(features, index)

        # 构建模型
        model = self.build_model()

        # 训练
        model.fit(x_train, y_train)
        # 预测
        y_predict = model.predict(x_test)
        # print(y_predict)
        # print(type(y_predict))
        # print(type(y_test))

        # 记录特征重要性
        features_importance = model.feature_importances_

        # 计算误差
        error_percentage = self.error_calculate(y_predict, y_test)

        return (features_importance, error_percentage, model)

    # 主程序，递归筛选特征，根据重要性筛选
    def choose_features(self, features):
        # 训练三次，将三次的误差取平均值，重要性取三次平均值

        features_importance = [0.0 for i in range(len(features))]
        error_percentage = 0.00
        # print(type(features))

        for i in range(3):
            temp_features_importance = []
            temp_error_percentage = 0.0

            (temp_features_importance, temp_error_percentage, model) = self.train(features, i)
            error_percentage += temp_error_percentage
            # print(temp_features_importance)
            for j in range(len(features_importance)):
                # print(features_importance[j])
                features_importance[j] += temp_features_importance[j]

        error_percentage = error_percentage / 3

        print(error_percentage)

        if self.flag == True:  # 将含有全部保留参数的模型也记录下来，该模型的训练结果不放到误差，重要性的统计数据中
            joblib.dump(model, self.save_path + "reserved\\" + self.name + "_reserved.pkl")
            file = open(self.save_path + "reserved\\" + "reserved_error.txt", mode="a+")
            self.reserved_min_error = error_percentage
            file.write(str(error_percentage))
            file.close()
            return

        for i in range(len(features_importance)):
            features_importance[i] = features_importance[i] / 3

        # 将特征和特征重要性拼到一起 格式如右 [('RM', 0.49359385750858875), ('LSTAT', 0.3256110013950264)]
        features_with_importance = list(zip(features, features_importance))
        """for i in features_importance:
            print(i)"""

        # 根据特征重要性进行排序，component[1]为重要性
        # 按降序排序
        features_with_importance = sorted(features_with_importance, key=lambda component: component[1], reverse=True)
        # print(features_with_importance)

        if self.min_error >= error_percentage:
            self.min_error = error_percentage
            self.final_features = [x[0] for x in features_with_importance]
            self.final_importance = [x[1] for x in features_with_importance]
            self.model = model
            joblib.dump(model, self.save_path + self.name + ".pkl")

        self.error_list.append(error_percentage)

        # print("error_percentage")
        # print(error_percentage * 100, "%")

        # 直到剩下的特征数小于等于left_num停止
        # sum_importance = sum(x[1] for x in features_with_importance)
        if len(features_with_importance) > self.left_num:
            # print("进入")
            f_length_ = len(features_with_importance)
            # 取除最不重要的step个
            features_with_importance = features_with_importance[0:f_length_ - self.step]

            new_features = [x[0] for x in features_with_importance]

            # 用剩下的特征进行下一次训练
            self.choose_features(new_features)
        else:
            # 递归终止,输出最终选的特征,并保存模型
            # joblib.dump(model,self.name+".pkl")
            # print("features_with_importance: "+name,features_with_importance)
            # error_list_global.extend(error_list)

            for feature in self.final_features:
                self.final_features_with_reserved.append(feature)
            for reserved_parameter in self.reserved_parameters:  # 将需要保留的参数强制加入到最后一轮训练中
                if reserved_parameter not in self.final_features:
                    self.final_features_with_reserved.append(reserved_parameter)

            self.flag = True
            self.choose_features(self.final_features_with_reserved)



    def plot_error(self):
        # x = [i for i in range(len(self.error_list))]
        # 画误差图，横坐标为剩余特征个数,纵坐标为误差
        x = [self.feature_length - self.step * i for i in range(len(self.error_list))]
        plt.plot(x, self.error_list)
        # plt.title("Error of " + self.name+"_"+self.prefix)
        plt.xlabel("The num of features")
        plt.ylabel("MAPE")
        plt.ylabel("MAPE")

        min_error_list_location = self.error_list.index(min(self.error_list))  # 误差最低点时训练的次数
        min_error = round(self.error_list[min_error_list_location], 3)
        min_error_location = len(self.init_features) - min_error_list_location * self.step  # 误差最低点剩下的参数个数

        min_error_round = round(min_error, 3)
        min_location = "(" + str(min_error_location) + ", " + str(min_error) + ")"
        plt.annotate(min_location, xytext=(min_error_location, min_error_round),
                     xy=(min_error_location, min_error))  # 在误差最低点加标注
        plt.tight_layout()
        plt.savefig(fname=self.save_path + self.name + "_" + self.prefix + "_error.png")

        plt.clf()

        for error in self.error_list:
            print(error, " ")
        print("\n")

    def plot_feature_importance(self):
        print("----------画图----------")
        print("\n")
        # x=[i for i in range(len(self.final_features))]

        sort_features = zip(self.final_features, self.final_importance)
        sort_features = sorted(sort_features, key=lambda component: component[1], reverse=True)

        features_ = [x[0] for x in sort_features]
        importances_ = [x[1] for x in sort_features]
        max_importance = max(importances_[0:10])
        min_importance = min(importances_[0:10])
        # 把特征重要性进行归一化,并画图
        importance_for_plot = [x / max_importance for x in importances_[0:10]]
        plt.bar(features_[0:10], importance_for_plot)

        plt.xticks(rotation=270)
        plt.xticks(fontsize=10)

        plt.xlabel("Features")
        plt.ylabel("Importance")
        # plt.figure(figsize=[30,60])
        plt.tight_layout()
        # plt.title(self.name+" important features")
        plt.savefig(fname=self.save_path + self.name + "_" + self.prefix + "_features.png")

        plt.clf()

    def features_to_file(self, str_time):
        # 将误差最低时的特征记录下来
        fileObject = open(self.save_path + self.prefix + "_features.txt", 'a+')
        fileObject.write('\n')
        fileObject.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        fileObject.write('\n')
        fileObject.write("训练时长")
        fileObject.write(str_time)
        fileObject.write('\n')
        # 算法名
        fileObject.write("使用算法: ")
        fileObject.write(self.name)
        fileObject.write('\n')
        # 最小误差
        float_error = float(self.min_error)
        str_error = str(float_error)
        fileObject.write(str_error)
        fileObject.write('\n')
        str_length = str(len(self.final_features))
        fileObject.write(str_length)
        fileObject.write('\n')
        sort_features = zip(self.final_features, self.final_importance)
        sort_features = sorted(sort_features, key=lambda component: component[1], reverse=True)

        # 写入特征
        for feature_importance_list in sort_features:
            fileObject.write(feature_importance_list[0] + ": ")
            fileObject.write(str(feature_importance_list[1]))
            fileObject.write('\n')
        fileObject.close();

    def error_to_file(self, str_time):
        # 将误差最低时的特征记录下来
        fileObject = open(self.save_path + self.prefix + "_error.txt", 'a+')
        fileObject.write('\n')
        fileObject.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        fileObject.write('\n')
        fileObject.write("训练时长")
        fileObject.write(str_time)
        # 算法名
        fileObject.write("使用算法: ")
        fileObject.write(self.name)
        fileObject.write('\n')
        # 最小误差
        float_error = float(self.min_error)
        str_error = str(float_error)
        fileObject.write(str_error)
        fileObject.write('\n')
        str_length = str(len(self.final_features))
        fileObject.write(str_length)
        fileObject.write('\n')
        # 写入误差
        for error in self.error_list:
            fileObject.write(str(error))
            fileObject.write('\n')
        fileObject.close()

        print("self.min_error" + str(self.min_error))
