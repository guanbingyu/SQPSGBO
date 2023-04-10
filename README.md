# SQPSGBO
Source code of "SQPSGBO:Low-Cost Big Data Performance Optimization of Spark SQL Applications"

## SensQuery folder

Code to search for CSQS

### run.sh
Automatically detect sensitive queries and output the number necessary to determine the CSQ in the $TOOL_HOME/SensQuery/result/ directory
According to. The script argument is (size, num)\

For example：
$TOOL_HOME/SensQuery/run.sh 21G 30\

The above command will probe tpcds-21G for CSQ. (The finish time used below is the end of the run time. )
The output in the $TOOL_HOME/SensQuery/result/TPCDS - 21 g - finishTime folder and the config file for detection sensitivity 
query the configuration to run during the process. The runtime is the execution time corresponding to each configuration.
query's Pearson correlation coefficient.csv, query's Spearman correlation coefficient.csv, Pearson correlation high Query.txt (correlation 
A coefficient greater than 0.75 is considered highly correlated, queries joined with '&' in each row are considered highly correlated), and Spearman's correlation is high Query.txt, query configuration for execution time-related statistics.csv, query configuration for execution time.csv\
 
### calculate.py

Calling this script computes the data related to the query and outputs the data necessary to judge the sensitive query.

Parameter Description：
For example：
python3 $TOOL_HOME/SensQuery/calculate.py -b=1 -e=30 -
p=$TOOL_HOME/data/runtime/tpcds/tpcds-21G -c=$TOOL_HOME/SensQuery/result/tpcds-21G
The above command will use $TOOL_HOME/data/runtime/TPCDS/TPCDS directory - 21 g execution time data to calculate the judgment Necessary to query data, its stored in $TOOL_HOME/SensQuery/result/TPCDS - 21 g

size：data size of tpcds,like 21G、300G.
num：30 is the recommended number of Settings to run to collect the data needed to determine a CSQ.

-b：The start sequence number of the execution time file number.
-e：The end sequence number of the execution time file number.
-p：The directory in which the execution time files are stored.
-c：The resulting output directory.

### result folder
Stores data necessary to judge sensitive queries


## SQPSGBO Method

Our proposed search method

### run.sh
The automated optimization script, which will run the automated optimization and output the results to $TOOL_HOME/CSGNet-BO/config 
Directory. Script parameters are (benchmark-size, type, and defalut_runtime)\


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
作用：对特定基准的配置参数进行寻优.
用法：python3 BO_DAF.py --benchmark=$1 --initpoints=$initNumber --
gan_initpoints=$ganinits --niters=$interationsNumber --csv_toconfig=$path/CSGNetConfig/ --
default_runtime=$3

参数说明：
benchmark-size：基准测试程序-数据集，例如wordcount-20G，redis-SS1_10
type：基准测试程序类型（wordcount和terasort填hibench，tpcds填tpcds，redis填redis）
defalut_runtime：默认配置执行时间（这个参数需要先手动跑默认配置得出执行时间后才能进行设
置）/

### config文件夹

存放调优算法过程中产生的配置文件和输出文件内容介绍：\
dataset- 存放配置失败后CSGNet根据当前最优配置生成的相似配置信息\
SnetConfig- CSGNet产生的所有初始样本配置文件\
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
生成相似样本并运行。\
target_space.py 运行当前的探索样本，将样本注册到hashset中，同时向返回样本的性能值和执行成
功/失败结果。\
util.py 使用高斯过程回归实现贝叶斯优化过程，根据当前已有样本空间选择下一个探索样本。\
event.py 存放当前bo搜索的状态。开始搜索、搜索过程、结束搜索。\
observer.py 观察当前event的状态进行通知logger执行对应操作。\
logger.py 将bo搜索过程的样本详细信息存入logs.json中。\
CSGNet.py 训练神经网络及生成配置\
model.py 神经网络结构\
Dataset.py 处理输入到神经网络的数据格式及构建训练集
