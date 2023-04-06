import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description=" ")
parser.add_argument('-b', '--begin', help='begin result num', type=int)
parser.add_argument('-e', '--end', help='end result num', type=int)
parser.add_argument('-p', '--path', help='result path', type=str)
parser.add_argument('-c', '--csv_path', help='csv path', type=str)

opts = parser.parse_args()

path = opts.path
begin = opts.begin
end = opts.end
csv_path = opts.csv_path

query = []  # 本次跑的query列表
LIST = range(begin, end + 1)  # 结果文件编号范围

# 为query列表赋值
lines = open(path + '/config' + str(begin) +'.result').readlines()
for line in lines:
    if "Time" in line:
        break
    query.append(line.split()[1])

# 获取执行结果
df = pd.DataFrame(columns=query, index=LIST, dtype='float64')
for i in LIST:
    if not os.path.exists(path + '/config' + str(i) +'.result'):
        continue
    lines = open(path + '/config' + str(i) +'.result').readlines()
    for line in lines:
        if '2000' in line:
            print('config'+str(i)+line)
        if 'Time' in line:
            break
        time = float(line.split()[0])
        name = line.split()[1]
        if time == 2000:
            df.loc[i].loc[name] = np.NAN
        else:
            df.loc[i].loc[name] = time

# 将0值替换
df = df.replace(0, np.nan)


# 计算pearson相关系数
corr = df.corr(method='pearson')
corr.to_csv(csv_path + "/query的pearson相关系数.csv")
fNew = open(csv_path + "/pearson相关性高的query.txt", "a+")
for i in range(0, len(query)):
    res = query[i]
    for j in range(0, len(query)):
        index = corr.loc[query[i]].loc[query[j]]
        if 0.75 < index < 1:
            res += '&&' + query[j]
    if len(res) > 14:
        fNew.write(res)
        fNew.write("\n")
fNew.close()

# 计算spearman相关系数
corr = df.corr(method='spearman')
corr.to_csv(csv_path + "/query的spearman相关系数.csv")
fNew = open(csv_path + "/spearman相关性高的query.txt", "a+")
for i in range(0, len(query)):
    res = query[i]
    for j in range(0, len(query)):
        index = corr.loc[query[i]].loc[query[j]]
        if 0.75 < index < 1:
            res += '&&' + query[j]
    if len(res) > 14:
        fNew.write(res)
        fNew.write("\n")
fNew.close()

# 输出执行时间表格
df = df.T
df.to_csv(csv_path + "/query配置对应执行时间" + ".csv")

# 计算统计量并输出表格
std_s = df.std(axis=1)
mean_s = df.mean(axis=1)
cv_s = std_s / mean_s
std_d = pd.DataFrame(std_s, columns=['std'])
mean_d = pd.DataFrame(mean_s, columns=['mean'])
mean_d = round(mean_d, 3)
cv_d = pd.DataFrame(cv_s, columns=['cv'])
res = pd.concat([mean_d, std_d, cv_d], axis=1)
print(res)
res.to_csv(csv_path + '/query配置对应执行时间相关统计量.csv')