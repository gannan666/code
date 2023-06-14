import pandas as pd
import numpy as np

def rw_csv():
    data=pd.read_csv('/home/l/下载/2023.csv', delimiter=',', skiprows=21018, nrows=60, encoding = 'gb2312')
    # head = data.head() # 查看csv文件头
    # columns = data.columns # 查看csv文件的列名称
    # null_all = data.isnull().sum() # 查看csv文件的空值
    # data = data.replace(' ',np.NaN) # 将csv文件的空格代替为空值
    # new_data = data.dropna(axis=0) # 将csv文件的空值全部删掉
    for row in data.values:
        if row[1] == 'det3d':
            with open("/home/l/下载/pcd.txt", 'a+') as f:
                f.writelines("tos://" + row[2]+'\n')