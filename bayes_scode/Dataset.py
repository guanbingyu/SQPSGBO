import pandas as pd
import numpy as np
import torch
def dataset(args,df,sparkConfRangeDf):
    samples, number_Features = df.shape
    '''
    2022/1/23
    存在异常值，筛选掉执行时间在后10%的样本异常值
    
    '''
    # print(sparkConfRangeDf)
    df=df.sort_values('runtime').reset_index(drop=True)
    df = df[:int(0.9 * samples)]
    samples=df.shape[0]
    mean = []
    mean1=[]
    mean2=[]
    index = []
    min = []
    max = []
    a = -1
    b = 1
    dataset = pd.DataFrame(df, copy=True)
    # 计算每列的平均值，最大值，最小值 #将数据映射到【-1，1】之间
    for indexs, row in df.iteritems():
        mean = np.append(mean, np.mean(row))
        index = np.append(index, indexs)
        min = np.append(min, np.min(row))
        max = np.append(max, np.max(row))
    '''
    2022/1/22  不再采用样本数据的最大值与最小值作为范围，而是采用实际范围
    '''
    #result存储所有数据的最大值与最小值
    results = pd.DataFrame(index=index,columns=['min','max','mean'])
    # print(results)
    results['min'] = min
    results['max'] = max
    results['mean'] = mean


    '''
    2022/1/22 
    筛选出 spark.memory.offHeap.size为零的数据,分别作为其的中位数，生成两种数据
    '''
    zero_data=df[df['spark.memory.offHeap.size']==0].reset_index(drop=True)
    no_zero_data=df[df['spark.memory.offHeap.size']!=0].reset_index(drop=True)

    for indexs, row in zero_data.iteritems():
        mean1=np.append(mean1,np.mean(row))
    for indexs, row in  no_zero_data.iteritems():
        mean2=np.append(mean2,np.mean(row))


    #spark.memory.offHeap.size为零的数据的均值
    mean1=mean1.reshape(1,args.number_features)
    mean_data1=pd.DataFrame(data=mean1,columns=index,copy=True)
    mean_data_n1=pd.DataFrame(data=mean1,columns=index,copy=True)

    #spark.memory.offHeap.size不为零的数据的均值
    mean2=mean2.reshape(1,args.number_features)
    mean_data2=pd.DataFrame(data=mean2,columns=index,copy=True)
    mean_data_n2=pd.DataFrame(data=mean2,columns=index,copy=True)

    #计算两种数据的比例
    ratio=zero_data.shape[0]/no_zero_data.shape[0]


    # print(results)
    i = 0
    dataset=pd.DataFrame(df,copy=True)

    '''
    2022/2/22
    按照样本的真实范围，处理原始数据
    改为两层循环
    '''
    for indexs, row in df.iteritems():

        for i in range(samples):
            try:
                if (indexs == 'runtime'):
                    #让处理后的配置能够找到跟
                    Y = a + (b - a) / (1.25*np.amax(row) - 0.75*np.min(row)) * (row[i] - 0.75*np.min(row))
                    runtime_max=1.25*np.amax(row)
                    runtime_min=0.75*np.min(row)

                else:
                    Y = a + ((b - a) / (sparkConfRangeDf.loc[indexs, 'max'] - sparkConfRangeDf.loc[indexs, 'min'])) * (
                            row[i] - sparkConfRangeDf.loc[indexs, 'min'])
                if(df.loc[[i],'spark.memory.offHeap.size'].item()==0)&(indexs=='spark.memory.offHeap.size'):
                    dataset.loc[[i],'spark.memory.offHeap.size'] = 0

                else:
                    dataset.loc[[i], indexs] = Y

            except KeyError:
                i = i + 1
                print("there are{} no config about {}".format(i, indexs))
    print(dataset.loc[:,'spark.memory.offHeap.size'])
    '''
    2022/2/22
    按照样本的真实范围，处理生成的均值数据
    '''
    #spark.memory.offHeap.size不为零的数据进行转换
    print(mean_data1)
    for indexs, row in mean_data1.iteritems():
        try:
            if (indexs == 'runtime'):
                Y = a + (b - a) / (runtime_max - runtime_min) * (row - runtime_min)
            else:
                Y = a + ((b - a) / (sparkConfRangeDf.loc[indexs, 'max'] - sparkConfRangeDf.loc[indexs, 'min'])) * (
                        row - sparkConfRangeDf.loc[indexs, 'min'])


            if (mean_data1['spark.memory.offHeap.size'].item()== 0) & (indexs == 'spark.memory.offHeap.size'):
                mean_data_n1['spark.memory.offHeap.size'] = 0
            else:
                mean_data_n1[indexs] = Y
        except KeyError:
            i = i + 1
            print("there are{} no config about {}".format(i, indexs))
    print(mean_data1.loc[:, 'spark.memory.offHeap.size'])
    print(mean_data_n1.loc[:, 'spark.memory.offHeap.size'])


    #spark.memory.offHeap.size不为零的数据进行转换
    print(mean_data2)
    for indexs, row in mean_data2.iteritems():
        try:
            if (indexs == 'runtime'):
                Y = a + (b - a) / (runtime_max - runtime_min) * (row - runtime_min)
            else:
                Y = a + ((b - a) / (sparkConfRangeDf.loc[indexs, 'max'] - sparkConfRangeDf.loc[indexs, 'min'])) * (
                        row - sparkConfRangeDf.loc[indexs, 'min'])
            mean_data_n2[indexs] = Y
        except KeyError:
            i = i + 1
            print("there are{} no config about {}".format(i, indexs))




    print(mean_data2.loc[:, 'spark.memory.offHeap.size'])
    print(mean_data_n2.loc[:, 'spark.memory.offHeap.size'])

    pd.DataFrame.to_csv(dataset, args.result_dir+args.realData_normalization)
    pd.DataFrame.to_csv(results, args.result_dir+args.real_MIN_MAX)
    pd.DataFrame.to_csv(mean_data1,args.result_dir+'mean_data1.csv')
    pd.DataFrame.to_csv(mean_data2, args.result_dir + 'mean_data2.csv')
    # 对数据进行归一化




    return dataset,mean_data1,mean_data_n1,mean_data2,mean_data_n2,ratio,results






def load_dataloader(dataset,batch_size,args):
    #转换为tensor数据类型
    data=torch.tensor(data=dataset.values.astype(float))
    data_shape=data.shape
    samples=int(data_shape[0]/batch_size)
    train_labels = torch.zeros((samples * batch_size, 1), device=torch.device(args.device))  # No of zeros
    train_set = [(data[i, :], train_labels[i]) for i in range(samples * batch_size)]
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
def load_2D_dataloader(dataset,batch_size,args):

    #转换为tensor数据类型,将一维数据转化为2维的数据
    data=torch.tensor(data=dataset.values.astype(float))
    data_shape=data.shape
    number=int(data_shape[0]/args.d_size)
    data=data[:number*args.d_size]
    data=data.reshape(-1,args.d_size,args.number_features)
    data = data.unsqueeze(1)
    data_shape = data.shape
    samples=int(data_shape[0]/batch_size)
    train_labels = torch.zeros((samples * batch_size, 1), device=torch.device(args.device))  # No of zeros
    train_set = [(data[i, :], train_labels[i]) for i in range(samples * batch_size)]
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)





#将10%的数据筛选掉
def data_processing (data):
    data=data.sort_values('runtime').reset_index(drop=True)
    number=len(data)
    number=int(0.9*number)
    data=data[:number+1]
    data=data.reset_index(drop=True)
    return data

def dataset_to_below_1(args,df,sparkConfRangeDf):
    # print(sparkConfRangeDf)
    a = -1
    b = 1
    # 计算每列的平均值，最大值，最小值 #将数据映射到【-1，1】之间
    i = 0
    dataset=pd.DataFrame(df,copy=True)
    for indexs, row in dataset.iteritems():
        try:
            Y = a + ((b - a) / (sparkConfRangeDf.loc[indexs, 'max'] - sparkConfRangeDf.loc[indexs, 'min'])) * (
                    row - sparkConfRangeDf.loc[indexs, 'min'])
            dataset[indexs] = Y
        except KeyError:
            i = i + 1
            print("there are{} no config about {}".format(i, indexs))

    return dataset




