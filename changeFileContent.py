import os

# 修改阈值脚本的 stop_time
def changeStopTime(file_path, stop_time):
    new_filename = os.path.splitext(file_path)[0] + '_bak.sh'
    old_filename = file_path
    print('修改的文件为 ' + file_path + '\t将stop_time修改为 ' + str(stop_time))

    # 打开旧文件
    f = open(old_filename,'r',encoding='utf-8')
    # 打开新文件
    f_new = open(new_filename,'w',encoding='utf-8')
    # 循环读取旧文件
    for line in f:
        if 'if ((runtime >' in line:
            target = line
            line = line.replace(target.split()[3].split(')')[0], str(stop_time))
            # 如果不符合就正常的将文件中的内容读取并且输出到新文件中
        f_new.write(line)
    f.close()
    f_new.close()


    # 获取当前文件路径
    current_path = os.path.abspath(file_path)
    # 获取当前文件的父目录,比如/usr/local/home/yyq/bo/rs_bo/rs_bo_newEI
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

    # new_fname = os.path.basename(new_filename)
    for file in os.listdir(father_path):
        if file == os.path.basename(new_filename):
            try:
                os.rename(new_filename, old_filename)
            except FileExistsError:
                os.remove(old_filename)
                os.rename(new_filename, old_filename)

def changeChmod755(stop_file_path):
    os.system('chmod 755 ' + stop_file_path)

if __name__ == '__main__':
    # file_path = '../shutDowonDetection-terasort.sh'
    file_path = 'E:\Desktop\ganrs_bo_test\shutDowonDetection-terasort.sh'
    stop_time = 50000
    changeStopTime(file_path, stop_time)
    changeChmod755(file_path)
