#!/bin/bash
#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount<3)); then
echo "no arg(benchmark-size, type, defalut_runtime)";
exit;
fi

initNumber=5
vsgnetinits=5
#bo迭代搜索50-initNumber次
interationsNumber=50

#获取benchmark_type (redis-RR1_10 => redis, wordcount => wordcount)
str=$1
OLD_IFS="$IFS" #保存旧的分隔符
IFS="-"
array=($str)
IFS="$OLD_IFS" # 将IFS恢复成原来的
benchmark=${array[0]}


path=$TOOL_HOME/CCAGNBO
echo $path

startTime=$(date "+%m-%d-%H-%M")
mv $path/config/$1 $path/config/$1-$startTime
mkdir -p $path/config/$1-$startTime
mv $path/logs_$benchmark.json $path/config/$1-$startTime
mv $path/generationConf.csv $path/config/$1-$startTime
mv $path/target.png $path/config/$1-$startTime
mv $path/GAN* $path/config/$1-$startTime
mv $path/general_data.csv $path/config/$1-$startTime
mv $path/sgan_sample.csv $path/config/$1-$startTime
mv $path/SnetConfig/ $path/config/$1-$startTime
mv $path/dataset/ $path/config/$1-$startTime
mv $path/output.txt $path/config/$1-$startTime
if [[ "$2" == "tpcds" ]]
then
        mv $TOOL_HOME/data/runtime/tpcds/$1 $TOOL_HOME/data/runtime/tpcds/tpcds-bo/$1-$startTime
	mkdir -p $TOOL_HOME/data/runtime/tpcds/$1
elif [[ "$2" == "redis" ]]
then
        mv $TOOL_HOME/data/runtime/redis/$1 $TOOL_HOME/data/runtime/redis/redis-bo/$1-$startTime
	mkdir -p $TOOL_HOME/data/runtime/redis/$1
fi

mkdir -p $path/config/$1
mkdir -p $path/SnetConfig/
mkdir -p $path/dataset/

python3 $path/VSGNet_Bayesian_Optimization.py --benchmark=$1 --initpoints=$initNumber --VSGNet_initpoints=$vsgnetinits --niters=$interationsNumber --csv_toconfig=$path/SnetConfig/ --default_runtime=$3

finishTime=$(date "+%m-%d-%H-%M")
mv $path/config/$1 $path/config/$1-$finishTime
mv $path/logs_$benchmark.json $path/config/$1-$finishTime
mv $path/generationConf.csv $path/config/$1-$finishTime
mv $path/target.png $path/config/$1-$finishTime
mv $path/GAN* $path/config/$1-$finishTime
mv $path/general_data.csv $path/config/$1-$finishTime
mv $path/sgan_sample.csv $path/config/$1-$finishTime
mv $path/SnetConfig/ $path/config/$1-$finishTime
mv $path/dataset/ $path/config/$1-$finishTime
mv $path/output.txt $path/config/$1-$finishTime
if [[ "$2" == "tpcds" ]]
then
        mv $TOOL_HOME/data/runtime/tpcds/$1 $TOOL_HOME/data/runtime/tpcds/tpcds-bo/$1-$finishTime
elif [[ "$2" == "redis" ]]
then
        mv $TOOL_HOME/data/runtime/redis/$1 $TOOL_HOME/data/runtime/redis/redis-bo/$1-$finishTime
fi
