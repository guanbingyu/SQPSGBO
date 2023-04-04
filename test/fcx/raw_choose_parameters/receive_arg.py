import argparse

import main_parameterChoose

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filePath',type=str,default='../gereator_data/WGAN_tpcds-21Greal0.5_generator.csv', help='Path of trainData')
parser.add_argument('-m', '--model_name',type=str, default='xgb',help='name of algorithm')
parser.add_argument('-s', '--save_path', type=str,default='../generart_oresult/tpcds-21G_gen/',help='path for saving files')
parser.add_argument( '--testdata',type=str,default='../gereator_data/WGAN_tpcds-21Grest0.5_real.csv', help='Path of testData')




parser.add_argument('-t', '--target', type=str,default='runtime',help='prediction target')  # 固定为 runtime
parser.add_argument('-step', '--step_nums',type=int,default=2, help='the num of parameters droped each time')
parser.add_argument('-left', '--left_nums',type=int,default=5, help='the num of parameters needed to be selected')
parser.add_argument('-n', '--name', type=str,default='parameters',help='parameters')  # 固定为 parameters
parser.add_argument('-q', '--quantile', default=0.9,help='the percentage of data used to train')
args = parser.parse_args()

filepath = args.filePath
name = args.name
save_path = args.save_path
target = str(args.target)
step = int(args.step_nums)
left_num = int(args.left_nums)
model_name = str(args.model_name)
quantile = args.quantile

model_namelist=['gbdt','lgb','rf','xgb','ada']

for model_name in model_namelist:
    '''
    加入test_data属性
    '''
    choose_main = main_parameterChoose.Main(test_data=args.testdata, file_path=filepath, save_path=save_path,
                                            model_name=model_name, step=step, left_num=left_num, name=name,
                                            quantile=quantile, target=target)



    choose_main.main()
