import argparse
import os
import pickle
import time

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from load_or_generate_data import load_or_generate_data_CLF
from sklearn import tree
# import graphviz

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_1', type=str, default=r'drone_meta_small/drone_audio_annotations.csv', help='root of data .csv')  # 存放数据集的位置，默认为'data/sudoku.csv'
parser.add_argument('--data_root_2', type=str, default=r'drone_audio_small', help='root of data .wave')  # 存放数据集的位置，默认为'data/sudoku.csv'
opt = parser.parse_args()  # 解析命令行参数，将用户输入的参数值赋给opt变量


# 调用函数，传入实际的文件路径
train_x, train_y = load_or_generate_data_CLF('CLF_features/CLF_features_train.pickle',root1=opt.data_root_1, root2=opt.data_root_2, type='train')
test_x, test_y = load_or_generate_data_CLF('CLF_features/CLF_features_test.pickle',root1=opt.data_root_1, root2=opt.data_root_2, type='test')

# 定义决策树模型，使用熵作为不纯度衡量标准
model = DecisionTreeClassifier(criterion='entropy', random_state=0)

# 使用训练数据训练模型
model.fit(train_x, train_y)

# 定义一个函数来绘制决策树
def visualize_tree(clf):
    # 增加图像尺寸和减小字体大小以减少重叠
    fig = plt.figure(figsize=(60, 40))
    plot_tree(clf, filled=True, rounded=True, fontsize=10)  # 减小字体大小
    plt.savefig('CLF_tree.png', dpi=300, bbox_inches='tight')
    # 可选：如果你想在保存后查看图像，可以注释掉上面的plt.savefig()后，运行plt.show()
    # plt.show()

# 调用函数绘制决策树
visualize_tree(model)

# 使用模型对测试集进行预测
print("开始训练")
start_time = time.time()  # 开始时钟记录时间
predictions = model.predict(test_x)
end_time = time.time()  # 结束时钟记录时间
run_time = end_time - start_time  # 计算运行时间
print(f"CLF训练时长: {run_time:.2f} 秒")

# 计算并打印准确率
print('准确率：', accuracy_score(test_y, predictions))

# 计算并打印混淆矩阵
confusion_matrix_results = confusion_matrix(test_y, predictions)
print('混淆矩阵：\n', confusion_matrix_results)