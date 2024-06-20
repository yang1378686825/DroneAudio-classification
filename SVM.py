import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from segdataset_trad import traditional_seg
import time
import pickle
import os
from load_or_generate_data import load_or_generate_data_SVM

# 调用traditional_seg函数加载和预处理数据，分割训练集和测试集
# 通常需要长达3min.
# print("开始加载数据集")
# start_time = time.time()  # 开始时钟记录时间
#
# train_x,train_y=traditional_seg(type='train')
# test_x,test_y=traditional_seg(type='test')
#
# end_time = time.time()  # 结束时钟记录时间
# run_time = end_time - start_time  # 计算运行时间
# print(f"加载数据集时长: {run_time:.2f} 秒")

# filename_x = f'SVM_features/SVM_x_features_{type}.pickle'
# filename_y = f'SVM_features/SVM_y_label_{type}.pickle'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_1', type=str, default=r'drone_meta_small/drone_audio_annotations.csv', help='root of data .csv')  # 存放数据集的位置，默认为'data/sudoku.csv'
parser.add_argument('--data_root_2', type=str, default=r'drone_audio_small', help='root of data .wave')  # 存放数据集的位置，默认为'data/sudoku.csv'
opt = parser.parse_args()  # 解析命令行参数，将用户输入的参数值赋给opt变量

# 调用函数，传入实际的文件路径
train_x, train_y = load_or_generate_data_SVM('SVM_features/SVM_features_train.pickle',root1=opt.data_root_1, root2=opt.data_root_2, type='train')
test_x, test_y = load_or_generate_data_SVM('SVM_features/SVM_features_test.pickle',root1=opt.data_root_1, root2=opt.data_root_2, type='test')

# 定义支持向量机模型参数，这里选择RBF核（默认），并设置正则化参数C为10，γ（核函数系数）为0.0001
# 注释中展示了尝试的不同模型参数，C=20, 1, 5, 10 的实验，最终选择了C=10作为最佳参数之一
# model = SVC(C=20.0, gamma=0.00001)#0.8
# model = SVC(C=1, gamma=0.00001)#0.675
# model = SVC(C=5, gamma=0.00001)#0.7875
model = SVC(C=10, gamma=0.00001)  # 0.825
# model = SVC(C=10)#0.8

# 用训练集训练：
print("开始训练")
start_time = time.time()  # 开始时钟记录时间
model.fit(train_x, train_y)  # 训练
end_time = time.time()  # 结束时钟记录时间
run_time = end_time - start_time  # 计算运行时间
print(f"SVM训练时长: {run_time:.2f} 秒")

# 用测试集预测：
prediction = model.predict(test_x)
print('准确率：', metrics.accuracy_score(prediction, test_y))
confusion = confusion_matrix(test_y, prediction)  # 计算混淆矩阵以评估分类性能
print(confusion)
