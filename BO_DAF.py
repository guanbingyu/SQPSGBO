import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import time
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from bayes_scode import JSONLogger, Events, BayesianOptimization
from bayes_scode.configuration import parser
import changeFileContent

args = parser.parse_args()
print('benchmark = ' + args.benchmark + '\t 初始样本数：initpoints = ' + str(
    args.initpoints) + '\t bo迭代搜索次数：--niters = ' + str(args.niters))

# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录,比如/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
# /usr/local/home/auto_tool
tool_home = os.getenv('TOOL_HOME')

# 服务器运行spark时config文件
config_run_path = father_path + "/config/" + args.benchmark + "/"
# 重要参数
vital_params_path = tool_home + "/common/parameters_set_" + args.benchmark.split('-')[0] + ".txt"
# 维护的参数-范围表
conf_range_table = tool_home + "/common/Spark_conf_range_" + args.benchmark.split('-')[0] + ".xlsx"
# 保存配置
generation_confs = father_path + "/generationConf.csv"


# --------------------- 生成 gan-rs 初始种群 end -------------------

def black_box_function(**params):
    i = []
    for conf in vital_params_name:
        i.append(params[conf])
    # print('black_box_function中schafferRun(i) 中的i为' + str(i))
    if 'redis' in args.benchmark:
        return schafferRun(i)

    return -schafferRun(i)


# 格式化参数配置：精度、单位等
def formatConf(conf, value):
    res = ''
    # 处理精度
    if confDict[conf]['pre'] == 1:
        res = round(value)
    elif confDict[conf]['pre'] == 0.01:
        res = round(value, 2)
    # 添加单位
    if not pd.isna(confDict[conf]['unit']):
        # 布尔值
        if confDict[conf]['unit'] == 'flag':
            res = str(bool(res)).lower()
        # 列表形式的参数（spark.serializer、spark.io.compression.codec等）
        elif confDict[conf]['unit'] == 'list':
            rangeList = confDict[conf]['Range'].split(' ')
            res = rangeList[int(res)]
        # 拼接上单位
        else:
            res = str(res) + confDict[conf]['unit']
    else:
        res = str(res)
    return res


all_results = []
# 1、单个配置 p写入到 /usr/local/home/zwr/hibench-spark-config/wordcount-100G-ga   命名：config1
# 2、run获取执行时间并返回
# 如果 cur_runtime == last_runtime 说明该配置执行失败
last_runtime = 1.0
benchmark_type = args.benchmark.split('-')[0]
unit = '' if benchmark_type == 'redis' else ' s'
print('基准为：' + benchmark_type + '  默认配置执行结果为 ' + str(args.default_runtime) + unit)
# 初始stop脚本的停止时间为默认配置的两倍
stop_file_path = tool_home + '/common/errorDetection/shutDowonDetection-' + benchmark_type + '.sh'

if benchmark_type == 'redis':
    # 超时停止时间固定为5分钟
    stop_time = 60 * 5
    changeFileContent.changeStopTime(stop_file_path, stop_time)
    changeFileContent.changeChmod755(stop_file_path)
else:
    changeFileContent.changeStopTime(stop_file_path, int(args.default_runtime * 2))
    changeFileContent.changeChmod755(stop_file_path)


def run(configNum):
    global last_runtime
    global max_runtime
    global stop_time
    # configNum = None
    # 使用给定配置运行基准测试程序
    run_cmd = tool_home + '/common/' + args.benchmark.split('-')[0] + '-ga.sh ' + str(
        configNum) + ' ' + father_path + '/config/' + args.benchmark + ' ' + args.benchmark.split('-')[1]
    print('configNum = ' + str(configNum) + '\t run_cmd = ' + str(run_cmd))
    if os.system(run_cmd) == 0:
        print('run_cmd命令执行成功')
    else:
        print('run_cmd命令执行失败')
    # 睡眠3秒，保证结果文件完成更新后再读取运行时间
    time.sleep(3)
    if benchmark_type == 'wordcount' or benchmark_type == 'terasort' or benchmark_type == 'join' or benchmark_type == 'scan' or benchmark_type == 'aggregation':
        # 获取此次spark程序的运行时间
        get_time_cmd = 'tail -n 1 $HIBENCH_HOME/report/hibench.report'
        line = os.popen(get_time_cmd).read()
        runtime = float(line.split()[4])
        # 配置失败了
        if runtime == last_runtime:
            # 失败的不是第一个配置。获取当前最优值，使最大阈值为最优值的10倍
            if len(all_results):
                # all_results 数组不为空
                runtime = int(min(all_results) * 10)
            else:
                # all_results 数组为空，第一个配置就失败了
                runtime = int(args.default_runtime * 2)
            all_results.append(runtime)
        # 该配置没有失败
        else:
            all_results.append(runtime)
            stop_time = int(min(all_results) * 5)
            changeFileContent.changeStopTime(stop_file_path, stop_time)
            changeFileContent.changeChmod755(stop_file_path)
            last_runtime = runtime
        return runtime
    elif benchmark_type == 'redis':
        # 超时停止时间固定为5分钟
        stop_time = 60 * 5
        changeFileContent.changeStopTime(stop_file_path, stop_time)
        changeFileContent.changeChmod755(stop_file_path)
        resultPath = tool_home + '/data/runtime/redis/' + args.benchmark + '/result' + str(configNum)
        # 如果路径里没有结果文件就说明执行失败
        is_find = os.path.exists(resultPath)
        # print('resultPath: ' + resultPath)
        # print('is_find: ' + is_find)
        if is_find == False:
            # 失败的不是第一个配置。获取当前最优值，使最大阈值为最优值的10倍
            if len(all_results):
                # all_results 数组不为空
                min_throughput = int(max(all_results) * 1 / 5)
            else:
                min_throughput = int(args.default_runtime * 1 / 2)
            all_results.append(min_throughput)
            return min_throughput
        # 从结果文件中获取吞吐量
        try:
            throughput = float(os.popen("head -n 13 " + resultPath + " | tail -n 1 | awk \'{print $2}\'").read())
            # print('throughput: ' + throughput)
            # 结果为0表示执行失败
            if throughput == 0.0:
                if len(all_results):
                    min_throughput = int(max(all_results) * 1 / 5)
                else:
                    min_throughput = int(args.default_runtime * 1 / 2)
                all_results.append(min_throughput)
                return min_throughput
            # 配置成功了
            else:
                all_results.append(throughput)
                return throughput
        except ValueError:
            pass
        if len(all_results):
            min_throughput = int(max(all_results) * 1 / 5)
        else:
            min_throughput = int(args.default_runtime * 1 / 2)
        all_results.append(min_throughput)
        return min_throughput
    elif benchmark_type == 'tpcds':
        resultPath = tool_home + '/data/runtime/tpcds/' + args.benchmark + '/config' + str(configNum) + '.result'
        # 如果路径里没有结果文件就说明执行失败
        is_find = os.path.exists(resultPath)
        if is_find == False:
            if len(all_results):
                runtime = int(min(all_results) * 10)
            else:
                runtime = int(args.default_runtime * 2)
            all_results.append(runtime)
            return runtime
        # 读取结果文件中的内容
        runtimes = open(resultPath).readlines()
        for i in range(0, len(runtimes)):
            # 执行成功返回时间
            if 'Time' in runtimes[i]:
                startTime = float(runtimes[i].split(':')[1])
                endTime = float(runtimes[i + 1].split(':')[1])
                runtime = endTime - startTime
                all_results.append(runtime)
                changeFileContent.changeStopTime(stop_file_path, int(min(all_results) * 5))
                changeFileContent.changeChmod755(stop_file_path)
                return runtime
            runtime = float(runtimes[i].split()[0])
            # 如果有一个query执行时间为2000则说明这个配置执行失败
            if runtime == 2000:
                if len(all_results):
                    runtime = int(min(all_results) * 10)
                else:
                    runtime = int(args.default_runtime * 2)
                all_results.append(runtime)
                return runtime
        return runtime

    elif benchmark_type == 'tpch':
        resultPath = tool_home + '/data/runtime/tpch/' + args.benchmark + '/config' + str(configNum) + '.result'
        # 如果路径里没有结果文件就说明执行失败
        is_find = os.path.exists(resultPath)
        if is_find == False:
            if len(all_results):
                runtime = int(min(all_results) * 10)
            else:
                runtime = int(args.default_runtime * 2)
            all_results.append(runtime)
            return runtime
        # 读取结果文件中的内容
        runtimes = open(resultPath).readlines()
        for i in range(0, len(runtimes)):
            # 执行成功返回时间
            if 'Time' in runtimes[i]:
                startTime = float(runtimes[i].split(':')[1])
                print(startTime)
                endTime = float(runtimes[i + 1].split(':')[1])
                print(endTime)
                runtime = endTime - startTime
                print(runtime)
                all_results.append(runtime)
                stop_time = int(min(all_results) * 5)
                changeFileContent.changeStopTime(stop_file_path, stop_time)
                changeFileContent.changeChmod755(stop_file_path)
                return runtime
            runtime = float(runtimes[i].split()[0])
            # 如果有一个query执行时间为2000则说明这个配置执行失败
            if runtime == 2000:
                if len(all_results):
                    runtime = int(min(all_results) * 10)
                else:
                    runtime = int(args.default_runtime * 2)
                all_results.append(runtime)
                return runtime
        return runtime


# 1、实际运行
configNum = 1


def schafferRun(p):
    global configNum
    # 打开配置文件模板
    fTemp = open(tool_home + '/common/configTemp_' + args.benchmark.split('-')[0], 'r')
    # 复制模板，并追加配置
    fNew = open(config_run_path + 'config' + str(configNum), 'a+')
    shutil.copyfileobj(fTemp, fNew, length=1024)
    try:
        for i in range(len(p)):
            if vital_params_name[i] == 'save1':
                fNew.write('  ')
                fNew.write('save')
                fNew.write('\t')
                fNew.write(formatConf(vital_params_name[i], p[i]))
                save2_index = vital_params_name.index("save2")
                fNew.write(' ')
                fNew.write(formatConf(vital_params_name[save2_index], p[save2_index]))
                fNew.write('\n')
            elif vital_params_name[i] == 'save2':
                pass
            else:
                fNew.write('  ')
                fNew.write(vital_params_name[i])
                fNew.write('\t')
                fNew.write(formatConf(vital_params_name[i], p[i]))
                fNew.write('\n')
    finally:
        fNew.close()
    runtime = run(configNum)
    configNum += 1
    #print(runtime)
    return runtime


def draw_target(bo):
    # 画图
    max = 0
    plt.plot(-bo.space.target, label='vsnet_bo  init_points = ' + str(init_points) + ', n_iter = ' + str(n_iter))
    if 'redis' in args.benchmark:
        max = bo._space.target.max()
    else:
        max = -bo._space.target.max()
    max_indx = bo._space.target.argmax()
    # 在图上描出执行时间最低点
    plt.scatter(max_indx, max, s=20, color='r')
    plt.annotate('maxIndex:' + str(max_indx + 1), xy=(max_indx, max), xycoords='data', xytext=(+20, +20),
                 textcoords='offset points'
                 , fontsize=12, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = .2'))
    plt.annotate(str(round(max, 2)) + 's', xy=(max_indx, max), xycoords='data', xytext=(+20, -20),
                 textcoords='offset points'
                 , fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad = .2'))
    plt.xlabel('iterations')
    plt.ylabel('runtime')
    plt.legend()
    plt.savefig(father_path + "/target.png")
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    # 读取重要参数
    vital_params = pd.read_csv(vital_params_path)
    print('重要参数列表（将贝叶斯的x_probe按照重要参数列表顺序转成配置文件实际运行:')
    print(vital_params)
    # 参数范围和精度，从参数范围表里面获取
    sparkConfRangeDf = pd.read_excel(conf_range_table)
    sparkConfRangeDf.set_index('SparkConf', inplace=True)
    confDict = sparkConfRangeDf.to_dict('index')

    # 遍历训练数据中的参数，读取其对应的参数空间
    d1 = {}
    d2 = {}
    for conf in vital_params['vital_params']:
        if conf in confDict:
            d1 = {conf: (confDict[conf]['min'], confDict[conf]['max'])}
            d2.update(d1)
        else:
            if conf == 'save':
                # save1
                conf1 = conf + '1'
                d1 = {conf1: (confDict[conf1]['min'], confDict[conf1]['max'])}
                d2.update(d1)
                # save2
                conf2 = conf + '2'
                d1 = {conf2: (confDict[conf2]['min'], confDict[conf2]['max'])}
                d2.update(d1)
            else:
                print(conf, '-----参数没有维护: ', '-----')

    # d2按照key值排序
    # print('按照key值排序前的d2 = ' + str(d2))
    sort_dict = {}
    for k in sorted(d2):
        sort_dict[k] = d2[k]
    d2 = sort_dict
    # print('按照key值排序后的d2 = ' + str(d2))

    vital_params_name = sorted(d2)
    # print('vital_params_name = ' + str(vital_params_name))
    vital_params_list = sorted(d2)
    vital_params_list.append('runtime')
    # print('vital_params_list = ' + str(vital_params_list))

    # 定义贝叶斯优化模型
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=d2,
        verbose=2,
        random_state=1,
        default_runtime=args.default_runtime,
        benchmark_type=benchmark_type
    )
    logpath = father_path + "/logs_" + benchmark_type + ".json"
    logger = JSONLogger(path=logpath)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # 随机样本个数
    init_points = args.initpoints
    n_iter = args.niters
    optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='combine')
    print(optimizer.max)
    draw_target(optimizer)

    # 存储数据
    # 读取json文件, 转成csv
    res_df = pd.DataFrame()
    for line in open(logpath).readlines():
        one_res = {}
        js_l = json.loads(line)
        if 'redis' in args.benchmark:
            one_res['target'] = js_l['target']
        else:
            one_res['target'] = -js_l['target']
        for pname in vital_params_name:
            one_res[pname] = js_l['params'][pname]
        df = pd.DataFrame(one_res, index=[0])
        res_df = res_df.append(df)
    # 设置索引从1开始
    res_df.index = range(1, len(res_df) + 1)
    res_df.to_csv(generation_confs)
