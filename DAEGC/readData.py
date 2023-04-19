import numpy as np

import pandas as pd
def read_csv_cols(filename, cols):
    # 读取CSV文件，指定要读取的列号
    df = pd.read_csv(filename, usecols=cols)
    # 将DataFrame对象转换为NumPy数组
    data = df.to_numpy()
    # 返回数组
    return data
# 读取两个CSV文件的数据
filename1 = '02癌症换特征加fid加编号.csv'
# filename2 = '02合并后的污染源加fid加编号.csv'
cols = [2, 3]  # 读取前两列作为二维坐标
data = read_csv_cols(filename1, cols)
# data1 = read_csv_cols(filename1, cols)
# data2 = read_csv_cols(filename2, cols)
np.save('./mydataset',data)