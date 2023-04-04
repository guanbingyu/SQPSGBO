import warnings
import pandas as pd
import numpy as np
import os
import time

from .target_space import TargetSpace
from .event import Events, DEFAULT_EVENTS
from .logger import _get_default_logger
from .util import UtilityFunction, acq_max, ensure_rng, acq_firstn
from .CSGNet import train
from .configuration import parser
args = parser.parse_args()

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录,比如/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI
father_path = os.path.abspath(os.path.dirname(os.path.dirname(current_path)) + os.path.sep + ".")

class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        self._queue.append(obj)


class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, default_runtime,benchmark_type,random_state=None, verbose=2,
                 bounds_transformer=None):
        """"""
        self._random_state = ensure_rng(random_state)

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self.benchmark_type = benchmark_type
        self.default_runtime = default_runtime
        self._space = TargetSpace(f, pbounds,benchmark_type, random_state)

        # queue
        self._queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)

    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
        else:
            # self._space.probe(params)
            # self.dispatch(Events.OPTIMIZATION_STEP)
            self.flag = True
            # success == True:配置执行成功，False 配置执行失败
            success, target = self._space.probe(params)
            if success:
                self.flag = True
            else:
                self.flag = False
            self.dispatch(Events.OPTIMIZATION_STEP)
            return self.flag, target
    def transfrom(self, data):
        a = -1
        b = 1
        train_X_temp = data
        for i in range(len(self._space.keys)):
            train_X_temp[:, i] = a + ((b - a) / (self._space.bounds[i][1] - self._space.bounds[i][0])) * (
                    train_X_temp[:, i] - self._space.bounds[i][0])

        return train_X_temp

    def transform_raw(self, data):
        a = -1
        b = 1
        train_X_temp = data
        i = 0
        for indexs, row in train_X_temp.iteritems():
            train_X_temp[indexs] = self._space.bounds[i][0] + ((train_X_temp[indexs] - a) *
                                                               (self._space.bounds[i][1] -
                                                                self._space.bounds[i][0])) / (b - a)
            i = i + 1
        return train_X_temp



    def suggest(self, utility_function):
        """Most promissing point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        # 忽略高斯过程中发出的警告信息
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            train_X = sc.fit_transform(np.array(self._space.params))
            print("训练集长度=", len(train_X))

            # 1. 使用目前已经搜索过的参数和target拟合高斯过程回归模型
            self._gp.fit(train_X, self._space.target)
            # print("sort = " + str(self._space.keys))

            # 2. 用于计算模型可信度
            self.computeConfidence(sc)

        # Finding argmax of the acquisition function.
        # 2. 找到采集函数的最大值，赋给 suggestion
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state,
            sc=sc
        )
        # 将suggestion封装成配置参数的样式传递回去做probe计算target值并注册到space中
        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        print('------------使用lhs生成初始样本点------------')
        lhsample = self._space.lhs_sample(init_points)
        for l in lhsample:
            self._queue.add(l.ravel())

    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def computeConfidence(self, sc): # 计算模型在测试样本上的模型精度，作为模型的可行度
        acc = self.accuracy(sc)
        self.confidence.append(acc)
        print("所有模型的可信度 ", self.confidence)

    # 生成测试集，用于计算每一轮GP模型的精度，从而确定其所占权重
    def getTestSamples(self):
        N = 5 # 最少有5个测试样本
        self.X_test, self.y_test = [], []
        while len(self.X_test) < N:
            X_test = self._random_state.uniform(self._space.bounds[:, 0], self._space.bounds[:, 1],
                                           size=(N, self._space.bounds.shape[0]))
            for x in X_test:
                params = dict(zip(self._space._keys, x))
                success, target = self._space.probe(params)
                if success == True:
                    self.X_test.append(x)
                    self.y_test.append(target)
                    print("生成测试集 - 配置成功 target", target, " conf", params)
                else:
                    print("生成测试集 - 配置失败 target", target, " conf", params)

    def accuracy(self, sc):
        x_test = sc.transform(self.X_test)
        y_test, r_std = self._gp.predict(x_test, return_std=True)
        acc = 0
        for i in range(len(y_test)):
            acc += np.abs(self.y_test[i] - y_test[i]) / np.abs(self.y_test[i])
        print("accuracy=", acc / len(y_test))
        return acc / len(y_test)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)
        # 记录搜索算法开始时间
        start_time = time.time()

        self.getTestSamples() # 生成测试集
        print("X_test", self.X_test, " y_test", self.y_test)

        self.confidence = [] # 模型的置信度为空

        self.ac_gains = {"ei":[0], "poi":[0], "ucb":[0]}  # 模型的置信度为空

        # 实例UtilityFunction，指定acq = ucb
        self.util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               confidence=self.confidence,
                               ac_gains=self.ac_gains,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)
        params_list = []
        params_list1 = []
        for param in self._space.keys:
            params_list.append(param)
            params_list1.append(param)

        # params_list.append('runtime')
        params_list1.append('runtime')
        # m 保存当前所有样本的df
        m = pd.DataFrame(columns=params_list)
        # 运行随机样本，保存在df m中

        while not self._queue.empty:
            x_probe = next(self._queue)
            success, target = self.probe(x_probe, lazy=False)
            # 获取该随机样本的执行时间
            x = self._space._as_array(x_probe)
            # 随机样本和执行时间存入m dataframe中
            sample = x.tolist()
            sample.append(target)
            # config.append(sample)
            print('初始采样：config\n' + str(sample))

            print(params_list1)
            n = pd.DataFrame(data=[sample], columns=params_list1)
            m = m.append(n, ignore_index=True)

        # 取随机样本中的最优样本 并训练GAN
        m = m.sort_values('runtime', ascending=False).reset_index(drop=True)
        bestconfig = m.iloc[:1, :-1]

        self.getBestSample_trainCSGNet(bestconfig, params_list, m, args.VSGNet_initpoints, params_list1)

        '''


        '''

        iteration = 0
        while iteration < n_iter:
            # print('key = \n' + str(self._space._keys))
            # print('bounds = \n' + str(self._space.bounds))
            # print('before probe, param.shape = ' + str(self._space.params.shape))
            # print('before probe, target = ' + str(self._space.target.shape))
            self.util.update_params()
            x_probe = self.suggest(self.util)
            iteration += 1
            # 该配置下执行成功，probe返回true
            success, target = self.probe(x_probe, lazy=False)
            if success:
                print('x_probe = ' + str(x_probe) + '\ntarget = ' + str(target))
            # 该配置执行失败，probe返回false，需要用网络生成一个样本
            else:
                iteration += 1
                print('该配置执行失败，需要使用gan生成一个样本 \t')
                params = self._space.params
                runtime = np.array([self._space.target]).T
                inits = np.hstack([params, runtime])
                train_df = pd.DataFrame(inits, columns=params_list1)
                # 取随机样本中的最优样本 并训练GAN
                train_df = train_df.sort_values('runtime', ascending=False).reset_index(drop=True)
                bestconfig = train_df.iloc[:1, :-1]

                target = self.getBestSample_trainCSGNet(bestconfig, params_list, train_df, 1, params_list1)

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))

        # 记录搜索算法结束时间
        end_time = time.time()
        print('算法搜索时间为： ' + str(int(end_time - start_time)) + 's')  # 秒级时间戳
        self.dispatch(Events.OPTIMIZATION_END)

    def getBestSample_trainCSGNet(self, bestconfig, params_list, m, num, params_list1):

        print('bestconfig\n' + str(bestconfig))
        # dataset 保存所有rs样本和gan样本
        dataset = pd.DataFrame(columns=params_list)

        first_time = time.time()
        generate_data = train(bestconfig, first_time, args)
        print(generate_data)

        # 利于适应度函数选择具体的样本点
        generate_data = self.transfrom(np.array(generate_data))
        # print(generate_data)
        generate_data = pd.DataFrame(data=generate_data, columns=params_list)
        # generate_data = self.suggest_n(utility_function=self.util, generate_data=generate_data)
        generate_data = generate_data[:num]
        # print(generate_data)
        # params_list.append('runtime')
        generate_data = self.transform_raw(generate_data)

        # 从GAN中选出第一个样本，并运行，保存在df m中
        print(num)
        for i in range(num):
            config = generate_data.iloc[i].tolist()
            # --------- 判断是否越界 ------------
            # print('参数和范围为\n' + str(self._space.keys) + "\n" + str(self._space.bounds))
            for i, bound in enumerate(self._space.bounds):
                # print('conf为:' + str(self._space.keys[i]) + ' 范围为 = ' + str(bound))
                if config[i] < bound[0]:
                    print(str(self._space.keys[i]) + "越界, 原值为 " + str(config[i]))
                    config[i] = bound[0]
                    print('越界处理后的值为 ' + str(config[i]))
                if config[i] > bound[1]:
                    print(str(self._space.keys[i]) + "越界, 原值为 " + str(config[i]))
                    config[i] = bound[1]
                    print('越界处理后的值为 ' + str(config[i]))
            # --------- 判断是否越界 ------------
            success, target = self.probe(config, lazy=False)
            # 获取该样本的执行时间
            try:
                config.append(target)
                print('GAN采样：config\n' + str(config))
                n = pd.DataFrame(data=[config], columns=params_list1)
                m = m.append(n, ignore_index=True)
            except KeyError:
                print('需要使用gan生成一个样本 \t')

        dataset = dataset.append(m, ignore_index=True)
        save_dataset = father_path + '/dataset'  # 保存网络的训练样本和生成样本
        if os.path.exists(save_dataset):
            pass
        else:
            os.mkdir(save_dataset)
        return target

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        self._gp.set_params(**params)
