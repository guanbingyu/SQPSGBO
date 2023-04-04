#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：bayes_opt 
@File ：LHS_sample.py
@Author ：Yang
@Date ：2021/12/29 18:30 
'''

import numpy as np

# D 参数个数， bounds= [[0,90],[0,30]]   N 需要生成几个初始样本
class LHSample():
    def __init__(self, D, bounds, N):
        self.D = D
        self.bounds = bounds;
        self.N = N

    def lhs(self):
        '''
        :param D:参数个数
        :param bounds:参数对应范围（list）
        :param N:拉丁超立方层数
        :return:样本数据
        '''
        # 结果为：N个样本，每个样本有D个参数（特征）
        result = np.empty([self.N, self.D])
        temp = np.empty([self.N])
        # 采样距离间隔
        d = 1.0 / self.N

        # D = 2， N = 30， i从0-1， j从0-29
        # 在【0，1】中间分成30个区域，在每个区域内生成一个实数，共生成30个实数存入temp[30]中
        # 对每一个参数，生成30个位于【0，1】之间有固定间隔的实数值
        for i in range(self.D):
            for j in range(self.N):
                temp[j] = np.random.uniform(
                    low=j * d, high=(j + 1) * d, size=1)[0]
            # 将序列的所有元素随机排序（打散temp数组）
            np.random.shuffle(temp)

            # 将temp中生成的30个实数赋值给result作为参数1的随机生成值
            for j in range(self.N):
                # 第j个样本，第i个参数值
                result[j, i] = temp[j]

        # 对样本数据进行拉伸
        b = np.array(self.bounds)
        # 获取所有参数的范围下界 [0 0]
        lower_bounds = b[:, 0]
        # 获取所有参数的范围上界 [90 30]
        upper_bounds = b[:, 1]
        # 如果下界超过上界的范围，报错
        if np.any(lower_bounds > upper_bounds):
            print('范围出错')
            return None

        #   sample * (upper_bound - lower_bound) + lower_bound
        # multiply数组和矩阵对应位置相乘：result = 30*2, (upper_bounds - lower_bounds) = 1*2扩展成30*2 ,对应位置相乘 返回30 * 2矩阵
        # add矩阵相加
        # print()
        np.add(np.multiply(result,
                           (upper_bounds - lower_bounds),
                           out=result),
               lower_bounds,
               out=result)
        return result


if __name__ == '__main__':
    D = 2 # 两个参数
    bounds = [[0,90],[0,30]]  # 参数的边界范围
    print(type(bounds))
    N = 30 # LHS层数为30层（将范围划分为30份）
    l = LHSample(len(bounds), bounds, N)
    result = l.lhs()
    print(result)


