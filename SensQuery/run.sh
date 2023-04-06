#D!/bin/bash
n3 calculate.py -b=1 -e=$2 -p=$TOOL_HOME/data/runtime/tpcds/tpcds-$1 -c=$path

#1 获取输入参数个数，如果没有参数，直接退出
pcount=$#
if((pcount<2)); then
echo "no arg(size, num)";
exit;
fi

# 建立配置存放目录
startTime=$(date "+%m-%d-%H-%M")
path=$TOOL_HOME/SensQuery/result/tpcds-$1-$startTime
mkdir -p $path/config

# 移动上轮剩余的结果文件
mv $TOOL_HOME/data/runtime/tpcds/tpcds-$1 $TOOL_HOME/data/runtime/tpcds/tpcds-$1-$startTime-lastRound
mkdir -p $TOOL_HOME/data/runtime/tpcds/tpcds-$1

# 生成配置
python3 $TOOL_HOME/common/configGenerate.py -cn=1 -cp=$path/config -p=$TOOL_HOME/common/parameters_set_tpcds.txt -n=$2 -t=tpcds -r=$TOOL_HOME/common/Spark_conf_range_tpcds.xlsx -a=true -m=8
export DATA_SIZE=$1
# 运行配置采集数据
for i in $(seq 1 1 $2)
do
	# 替换配置
	rm -rf $SPARK_HOME/conf/spark-defaults.conf
	cp $path/config/config$i $SPARK_HOME/conf/spark-defaults.conf
	
	# 开启spark程序 
	echo ================= config$i =================
	echo $(date)
	source $TOOL_HOME/common/tpcds/tpcds-env.sh
	python3 $TOOL_HOME/common/tpcds/tpcdsMulti.py 4 tpcds-$1 $i False
done

python3 calculate.py -b=1 -e=$2 -p=$TOOL_HOME/data/runtime/tpcds/tpcds-$1 -c=$path
mv $TOOL_HOME/data/runtime/tpcds/tpcds-$1 $path/runtime
