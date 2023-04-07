# SQPSGBO
Source code of "SQPSGBO:Low-Cost Big Data Performance Optimization of Spark SQL Applications"


### run.sh
自动化优化脚本，调用该脚本将进行自动化优化，最终将结果输出到$TOOL_HOME/VSGNet-BO/config
目录下。脚本参数为(benchmark-size, type, defalut_runtime)\
\
示例：
$TOOL_HOME/SQPSGBO/run.sh wordcount-20G hibench 257.824\
上述命令将优化wordcount-20G。（下面用到的finishTime代表运行结束时间）输出结果在
$TOOL_HOME/SQPSGBO/config/wordcount-20G-finishTime/目录中，dataset存放配置失败后
VSGNet根据当前最优配置生成的相似配置信息，SnetConfig存放VSGNet产生的所有初始样本配置文
件，generationConf.csv存放搜索过程中的所有样本配置参数信息，logs.json记录bo算法的输出数据，
存放搜索过程中每个样本的相关信息，output.txt记录失败配置的信息，如果该文件为空表示搜索过程
中没有失败配置，target.png描绘搜索样本性能提升过程，横坐标对应样本的迭代次数，纵坐标对应性
能指标，config*为每个样本对应的配置文件

### changeFileContent.py
作用：修改杀死超时配置中的超时时间（以 s 为单位）
用法：
file_path = '$TOOL_HOME\common/errorDetection\shutDowonDetection-terasort.sh'
stop_time = 50000
changeStopTime(file_path, stop_time)
changeChmod755(file_path)\

### BO_DAF.py
作用：对特定基准的配置参数进行寻优，拉丁超立方采样3个初始样本和VSGnet生成3个初始样本（共6
个初始样本），使用贝叶斯和VSGnet迭代寻优的方式搜索该基准的最优配置参数，若当前配置运行成功
则继续使用贝叶斯优化获取下一探索样本，并把kill超时配置的超时时间改为最优配置的5倍。若配置运
行失败则将配置性能设置为最优配置的10倍（redis则设置为最优配置的1/5），同时选择下一探索样本
时不采用高斯过程回归而是使用VSGnet根据最优样本生成一个近似样本作为下一探索样本，VSGnet通
过这种方式指导贝叶斯优化的搜索过程。若当前配置的优化配置是默认配置的5倍则提前停止搜索，否则
直到满足指定的搜索次数为止。/
用法：python3 ganinbo_Bayesian_Optimization.py --benchmark=$1 --initpoints=$initNumber --
gan_initpoints=$ganinits --niters=$interationsNumber --csv_toconfig=$path/SnetConfig/ --
default_runtime=$3/
参数说明：
benchmark-size：基准测试程序-数据集，例如wordcount-20G，redis-SS1_10
type：基准测试程序类型（wordcount和terasort填hibench，tpcds填tpcds，redis填redis）
defalut_runtime：默认配置执行时间（这个参数需要先手动跑默认配置得出执行时间后才能进行设
置）/

### config文件夹

存放调优算法过程中产生的配置文件和输出文件内容介绍：\
dataset- 存放配置失败后VSGNet根据当前最优配置生成的相似配置信息\
SnetConfig- VSGNet产生的所有初始样本配置文件\
generationConf.csv 存放搜索过程中的所有样本配置参数信息\
logs.json bo算法的输出数据，存放搜索过程中每个样本的相关信息\
output.txt 记录失败配置的信息，如果该文件为空表示搜索过程中没有失败配置\
target.png 描绘搜索样本性能提升过程，横坐标对应样本的迭代次数，纵坐标对应性能指标\
confign 每个样本对应的配置文件


### bayes_scode文件夹
BO_DAF.py 搜索过程中调用的辅助文件\
内容介绍：\
configuration.py 所有文件中使用的超参数都统一存放在这个文件中。\
LHS_sample.py 根据参数个数和范围进行拉丁超立方采样。\
bayesian_optimization.py 创建贝叶斯优化和高斯过程对象。使用VSGnet生成初始样本。根据
target_space.py返回值判断当前配置成功/失败，配置成功则继续探索，否则根据最优样本使用VSGnet
生成相似样本并运行。判断当前配置的优化到达5倍则停止搜索。\
target_space.py 运行当前的探索样本，将样本注册到hashset中，同时向返回样本的性能值和执行成
功/失败结果。\
util.py 使用高斯过程回归实现贝叶斯优化过程，根据当前已有样本空间选择下一个探索样本。\
event.py 存放当前bo搜索的状态。开始搜索、搜索过程、结束搜索。\
observer.py 观察当前event的状态进行通知logger执行对应操作。\
logger.py 将bo搜索过程的样本详细信息存入logs.json中。\
CSGNet.py 训练神经网络及生成配置\
model.py 神经网络结构\
Dataset.py 处理输入到神经网络的数据格式及构建训练集
