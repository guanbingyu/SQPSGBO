import warnings
import numpy as np
from scipy.stats import norm
import random
from scipy.optimize import minimize
import time

i = 1
def logFile(content): # 将信息存入文件
    fp = open('argmax.txt', 'a', encoding='utf-8')
    fp.write(content)
    fp.write('\n')  # 换行
    fp.close()

def acq_max(ac, gp, y_max, bounds, random_state, sc, n_warmup=10000, n_iter=10):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    global i

    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))

    ys = ac(x_tries, gp=gp, y_max=y_max, sc=sc)
    print("ys", ys)

    if len(set(ys)) == 1:
        maxindex = np.random.randint(0, len(ys), 1)
        logFile(str(i) + " 所有候选样本的均值方差完全相同，随机选点 ys.argmax() " + str(maxindex))  # 将每轮选择的样本下标存入文件中
        print(i, "所有候选样本的均值方差完全相同，随机选点 ys.argmax()", maxindex)
    else:
        maxindex = ys.argmax()
        logFile(str(i) + " ys.argmax() " + str(maxindex))  # 将每轮选择的样本下标存入文件中
        print(i, " ys.argmax()", maxindex)



    i += 1

    # argmax最大值所对应的索引， 取出预测的最大y值对应的样本x
    x_max = x_tries[maxindex]
    # 记录采集函数找到的最大样本点，用于explore样本点时的更新使用
    max_acq = ys.max()


    # # Explore the parameter space more throughly
    x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_iter, bounds.shape[0]))
    # explore 随机产生n_iter个（10个）样本点，看看能不能找到更好的样本点赋给x_max

    for x_try in x_seeds:
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max, sc=sc),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        if not res.success:
            continue

        # 如果找到了比max_acq还要大的样本x，则更新max_acq
        if max_acq is None or -res.fun >= max_acq:
            x_max = res.x
            max_acq = -res.fun

    # np.clip将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min\
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])
'''
2022/4/15
选择前n适应度值的配置
'''

def acq_firstn(ac, gp, y_max,generate_data):




    # --------------------------------------------------------
    test_X_temp = np.array(generate_data)
    # print(test_X_temp)
    upper = ac(x=test_X_temp, gp=gp, y_max=y_max)
    print('snet生成配置的置信度')
    print(upper)
    # print('upper = \n' + str(upper))
    # upper 从大到小排序
    generate_data['acq_value']=upper

    generate_data=generate_data.sort_values('acq_value',ascending=False).reset_index(drop=True)
    generate_data=generate_data.drop('acq_value', axis=1)
    return generate_data


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """
    # kind : 采集函数的类型（ucb、ei、poi）
    def __init__(self, kind, kappa, xi, confidence, ac_gains, kappa_decay=1, kappa_decay_delay=0):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi
        
        self._iters_counter = 0

        self.confidence = confidence # 模型的置信度(accuracy)
        self.ac_gains = ac_gains # 历史增益

        self.af = ["ei", "poi", "ucb"]


        if kind not in ['ucb', 'ei', 'poi', 'combine', 'ts']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max, sc):
        x = sc.transform(x)
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)
        if self.kind == 'ts':
            return self._ts(x, gp)
        if self.kind == 'combine':
            return self._combineAC(x, gp, y_max, self.xi, self.kappa, sc)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std


    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
  
        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)

    @staticmethod
    def _ts(x, gp): # 汤普森抽样
        # with warnings.catch_warnings():

        posterior_sample = gp.sample_y(x, 1).T[0]
        # print("posterior_sample", posterior_sample)
        return posterior_sample

    choosed_acq = 0
    
    
    def _combineAC(self, x, gp, y_max, xi, kappa, sc):
        global choosed_acq

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        EI = self._ei(x, gp, y_max, xi)
        POI = self._poi(x, gp, y_max, xi)
        UCB = self._ucb(x, gp, kappa)
        acqs_values = {0: EI, 1: POI, 2: UCB}

        if x.shape[0] == 1: # 只有一个候选样本x，表示是x_seed点
            # print("X_seeds")
            return acqs_values[choosed_acq]

        max_idx = [EI.argmax(), POI.argmax(), UCB.argmax()]
        # 从三个获取函数中分别选出各自的候选点
        print('三个采集函数选出的max_idx', max_idx)
        logFile('三个采集函数选出的max_idx ' + str(max_idx))

        #计算三个最好候选点上的增益
        max_points = [x[max_idx[0]], x[max_idx[1]], x[max_idx[2]]]
        max_means,max_stds=gp.predict(max_points, return_std=True)
        print("max_means", str(max_means), " max_stds", max_stds)

        self.model_reliability() # 计算模型可信度
        self.acculativeGain(max_means, max_stds, y_max) # 计算三个采集函数的累积增益
        print("三个函数的累计增益为" + str(self.ac_acculativeGain))

        total = np.sum(self.ac_acculativeGain)
        possibility = [acgain / total for acgain in self.ac_acculativeGain]
        # print("选择概率为" + str(possibility))
        idx = self.whichOne(possibility)  # 按照概率选择候选点下标
        choosed_acq = idx # 记录本轮选择的AC，用于X_seed
        return acqs_values[idx] # 返回选择的采集函数在x上的结果值


    def acculativeGain(self, mu, sigma, ymax): # 计算本轮累计增益，用于计算AC的选取概率
        self.ac_acculativeGain = []
        cur_gain = self.computeGain(mu, sigma, ymax) # 计算本轮增益
        for i, ac in enumerate(self.af):
            his_gain = self.ac_gains[ac][:-1]
            temp = np.multiply(np.array(his_gain),np.array(self.reliability))
            his_gain_res = np.sum(temp.tolist())

            acculative_gain = cur_gain[i] + his_gain_res
            self.ac_acculativeGain.append(acculative_gain)


    def model_reliability(self):  # 计算每个模型在所有模型中的可信度
        self.reliability = []
        sum = np.sum(self.confidence)
        for c in self.confidence:
            self.reliability.append(c / sum)
        print("所有模型的可信度所占权重 ", self.reliability)

    def computeGain(self, mu, sigma, ymax): # 计算本轮增益（不包含历史增益）
        cur_gain = []
        for i, m in enumerate(zip(mu, sigma)):
            temp = (m[0] - ymax) / m[1]
            gain = norm.cdf(temp) # 计算本轮增益
            self.ac_gains[self.af[i]].append(gain) # 把本轮增益加入历史增益
            cur_gain.append(gain)
        # print("本轮增益加入历史增益", cur_gain, " 历史增益", self.ac_gains)
        return cur_gain

    # 轮盘赌主函数：共开启T轮轮盘赌，返回选择的下标
    def whichOne(self, probability):
        T = 100
        result = np.zeros(T).astype(int)
        count = np.zeros(len(probability)).astype(int)
        print("probability", probability)

        probabilityTotal = np.zeros(len(probability))
        probabilityTmp = 0
        for i in range(len(probability)):
            probabilityTmp += probability[i]
            probabilityTotal[i] = probabilityTmp
        print("probabilityTotal", probabilityTotal)

        for i in range(T):  # 开启T轮轮盘赌
            result[i] = self.roulette(probabilityTotal)
            count[result[i]] += 1
        print("每一轮选择的下标，result:", result)
        print("每个概率被转中的次数, count:", count)
        chooseIdx = self.whichOneHelper(count)
        print("choose which one?", [chooseIdx])
        return chooseIdx # 返回T轮轮盘赌后，选中次数最多的下标

    def whichOneHelper(self, count):  # 当最多选中的AC下标不止一个时
        max = np.max(count)  # 记录最多的选中次数
        maxIndex = []  # 记录选中次数最多的下标
        for index, nums in enumerate(count):
            if nums == max:
                maxIndex.append(index)

        for i in range(20):
            if len(maxIndex) == 1:  # 选中次数最多的元素只有一个
                return np.argmax(count)
            elif len(set(count)) == 2:  # 选中次数最多的元素有两个
                r = np.random.randint(0, 2, 99).tolist()  # 产生99个0-1之间的随机整数
                dic = {i: r.count(i) for i in r}  # 分别统计0和1出现的次数
                x = sorted(dic.items(), key=lambda x: x[1], reverse=True)[0][0]  # 按照次数降序排序
                return maxIndex[x]

    # 辅助函数：用于开启一轮的轮盘赌，返回当前轮数选中的下标
    def roulette(self, probabilityTotal):
        randomNumber = np.random.rand()
        result = 0
        for i in range(1, len(probabilityTotal)):
            if randomNumber < probabilityTotal[0]:
                result = 0  # 选中第一个
                # print("choose", [result], "random number:", randomNumber, "< index 0:", probabilityTotal[0])
                break
            elif probabilityTotal[i - 1] < randomNumber <= probabilityTotal[i]:
                result = i  # 选中第i个
                # print("choose", [result], "index ", i - 1, ":", probabilityTotal[i - 1], "< random number:",
                #       randomNumber, "< index ", i, ":", probabilityTotal[i])
        return result



def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)
