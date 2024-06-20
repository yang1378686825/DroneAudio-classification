import argparse
import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import seaborn as sn
from sklearn import model_selection
from sklearn import preprocessing
import IPython.display as ipd
# define directorie
# load a wave data
from sklearn.model_selection import train_test_split


def show_wave(x):
    plt.plot(x)
    plt.show()

# 展示音频信号的梅尔频谱(Mel Spectrogram)热力图
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)  # melsp通常是一个二维数组，代表 Mel 频谱数据
    plt.colorbar()
    plt.show()

# 添加白噪声：数据增强
def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))

# 数据增强：时间帧偏移声音 shift sound in timeframe
def shift_sound(x, rate=3):
    return np.roll(x, int(len(x)//rate))


def read_file_list(root1=r"meta/esc50.csv", root2=r"audio", type='train', random_state=1):
    """
    从CSV文件和指定音频目录读取元数据，并根据指定类型分割数据集为训练集或测试集。

    参数:
    - root1: 元数据CSV文件路径，默认为 "meta/esc50.csv"
    - root2: 音频文件根目录，默认为 "audio"
    - type: 分割类型，'train' 或 'test'，默认为 'train'
    - random_state: 随机种子，保证数据分割的可复现性，默认为 1

    返回:
    - 根据 type 参数返回对应的文件路径列表和标签列表
    """
    # 读取元数据CSV文件
    meta_data = pd.read_csv(root1)

    # 直接提取文件名和目标标签
    name = meta_data["filename"].tolist()
    label = meta_data["target"].tolist()  # 假设目标标签就是我们需要的，无需转换

    # 构建音频文件的完整路径列表
    path = [os.path.join(root2, filename) for filename in name]

    # 利用train_test_split函数分割数据集
    signals_train, signals_test, labels_train, labels_test = train_test_split(path, label, test_size=0.2, random_state=random_state)

    # 根据传入的 type 参数返回对应的数据集部分
    if type == 'train':
        return signals_train, labels_train
    elif type == 'test':
        return signals_test, labels_test
    else:
        raise ValueError("Invalid type specified. Please use either 'train' or 'test'.")




# 提取 (MFCCs) 特征及其对应的标签
def traditional_seg(root1=r"meta/esc50.csv", root2=r"audio", type='train', random_state=1):
    """
    从音频文件中提取梅尔频率倒谱系数 (MFCCs) 特征，并返回特征及其对应的标签。

    参数:
    - root1: 元数据文件路径，默认为 "meta/esc50.csv"，包含音频文件路径和标签信息。
    - root2: 音频文件存放的根目录，默认为 "audio"。
    - type: 数据集类型，'train' 表示训练集，'test' 表示测试集，默认为 'train'。
    - random_state: 随机种子，用于确保数据分割的可复现性，默认为 1。

    返回:
    - mfccs: 所有音频文件的 MFCC 特征列表。
    - labels: 对应音频文件的标签列表。
    """
    # 读取音频文件路径和标签
    signals, labels = read_file_list(root1=root1, root2=root2, type=type, random_state=random_state)

    # 初始化 MFCC 特征列表
    mfccs = []

    # 遍历音频文件路径列表
    for signal in signals:
        # 加载音频文件，设定采样率为 44100 Hz
        x, fs = librosa.load(signal, sr=44100)

        # 提取单个 MFCC 特征，通常 n_mfcc>1 以获得更多系数，这里简化处理取 n_mfcc=1
        mfcc = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=1).squeeze()  # squeeze 用于去掉可能的单维度

        # 将当前音频的 MFCC 特征添加到列表中
        mfccs.append(mfcc)

    # 返回 MFCC 特征列表和对应的标签列表
    return mfccs, labels




#############################################
# 定义数据集类，继承自torch的Dataset，返回mfcc特征和标签
class SegDataset_rnn(torch.utils.data.Dataset):
    def __init__(self, root1=r"meta/esc50.csv", root2=r"audio", type='train', random_state=1):
        """
        初始化数据集
        :param root1: 元数据文件路径，默认指向ESC-50数据集的元数据CSV文件
        :param root2: 音频文件存放目录的根路径，默认为"audio"
        :param type: 数据集类型，'train'表示训练集，'test'表示测试集，默认为'train'
        :param random_state: 随机种子，用于数据划分的一致性
        """
        # 调用外部函数read_file_list读取音频文件路径和标签列表
        signals, labels = read_file_list(root1=root1, root2=root2, type=type, random_state=random_state)
        # 将信号和标签分别存储为数据集的成员变量
        self.signals = signals
        self.labels = labels

        # 打印出数据集中有效样本的数量
        print('Read ' + str(len(self.labels)) + ' valid examples')

    # 定义一个生成随机浮点数的方法，常用于随机决策
    def rand(self, a=0, b=1):
        """
        生成一个位于区间[a, b)的随机浮点数
        """
        return np.random.rand() * (b - a) + a

    # 重写__getitem__方法，用于根据索引获取数据集中的单个样本
    def __getitem__(self, idx):
        """
        :param idx: 索引值
        :return: 一个样本MFCCs特征和标签
        """
        # 根据索引获取信号路径和对应的标签
        signal = self.signals[idx]
        label = self.labels[idx]

        # 使用librosa加载音频文件，设定采样率为44100Hz
        # x, fs = librosa.load(signal, sr=44100)
        x, fs = librosa.load(signal, sr=16100)

        # # 数据增强部分：随机决定是否进行噪声添加和时间偏移
        # # 噪声添加，50%的概率
        # if self.rand() < 0.5:
        #     x = add_white_noise(x)  # 添加白噪声
        #
        # # 时间偏移，也是50%的概率
        # if self.rand() < 0.5:
        #     x = shift_sound(x)  # 时间偏移音频

        # 计算音频的MFCC特征
        # 函数返回一个二维数组，形状为 (n_mfcc, t)，其中 n_mfcc=20 是用户指定的 MFCC 维度，t=32或87 表示时间帧的数量, 即seq_length
        mfcc = librosa.feature.mfcc(y=x, sr=fs, S=None, n_mfcc=20)

        # 调整MFCC的维度顺序以适配模型输入
        mfcc = mfcc.swapaxes(0, 1)  # 变为431*20

        # 将numpy数组转换为PyTorch的张量，类型为FloatTensor
        mfcc = torch.from_numpy(mfcc).type(torch.FloatTensor)

        # 将标签也转换为PyTorch张量
        label = torch.tensor(label)

        # 返回mfcc特征和标签
        return mfcc, label
#############################################

    def __len__(self):
        return len(self.labels)



#############################################
# 定义数据集类，返回综合特征和标签，用于多层感知器(MLP)模型的输入
class SegDataset_mlp(torch.utils.data.Dataset):
    def __init__(self, root1=r"meta/esc50.csv", root2=r"audio", type='train', random_state=1):
        """
        初始化数据集，读取音频文件路径及其标签。

        参数:
        - root1: 元数据CSV文件路径，默认为"meta/esc50.csv"
        - root2: 音频文件根目录，默认为"audio"
        - type: 数据集类型，如'train'或'val'，默认为'train'
        - random_state: 随机种子，默认为1
        """
        signals, labels = read_file_list(root1=root1, root2=root2, type=type, random_state=random_state)
        self.signals = signals  # 存储音频文件路径列表
        self.labels = labels  # 存储对应标签列表
        print(f'Read {len(self.labels)} valid examples')

    def rand(self, a=0, b=1):
        """生成[a, b)范围内的随机浮点数"""
        return np.random.rand() * (b - a) + a

    def __getitem__(self, idx):
        """
        通过索引获取单个样本，包括预处理音频并提取特征。

        参数:
        - idx: 索引值

        返回:
        - features: 综合音频特征的张量
        - label: 对应的标签张量
        """
        signal_path = self.signals[idx]  # 获取音频文件路径
        label = self.labels[idx]  # 获取对应标签

        # 使用librosa加载音频文件
        # X, sample_rate = librosa.load(signal_path, sr=44100)
        X, sample_rate = librosa.load(signal_path, sr=16100)
        # 下面的代码块是数据增强，当前已被注释掉
        # noise = self.rand() < .5
        # shift = self.rand() < .5
        # if noise:
        #     X = add_white_noise(X)  # 添加白噪声
        # if shift:
        #     X = shift_sound(X)  # 时间偏移

        # 提取音频特征
        stft = np.abs(librosa.stft(X))  # 短时傅里叶变换的绝对值
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13).T, axis=0)  # 和SegDataset不同的是这里计算了所有帧的mfcc均值，最后只输出13个参数
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)  # 色度特征
        mel = np.mean(librosa.feature.melspectrogram(S=stft, n_mels=13, sr=sample_rate).T, axis=0)  # 梅尔频谱图
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)  # 频谱对比
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)  # Tonnetz特征

        # 合并所有特征
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        # 转换为PyTorch张量
        features = torch.from_numpy(ext_features).type(torch.FloatTensor)
        label = torch.tensor(label)

        return features, label  # 返回综合特征和标签
#############################################

    def __len__(self):
        return len(self.labels)



#############################################
# 定义数据集类，继承自torch的Dataset，返回特征和标签
class SegDataset_transformer(torch.utils.data.Dataset):
    def __init__(self, root1=r"meta/esc50.csv", root2=r"audio", type='train', random_state=1):

        # 调用外部函数read_file_list读取音频文件路径和标签列表
        signals, labels = read_file_list(root1=root1, root2=root2, type=type, random_state=random_state)
        # 将信号和标签分别存储为数据集的成员变量
        self.signals = signals
        self.labels = labels

        # 打印出数据集中有效样本的数量
        print('Read ' + str(len(self.labels)) + ' valid examples')

    # 重写__getitem__方法，用于根据索引获取数据集中的单个样本
    def __getitem__(self, idx):

        signal = self.signals[idx]
        label = self.labels[idx]

        x, sr = librosa.load(signal, sr=16100)

        # 计算音频的MFCC极其一阶二阶特征和其他特征
        # 函数返回一个二维数组，形状为 (n_mfcc, t)，其中 n_mfcc=20 是用户指定的 MFCC 维度，t=32或87 表示时间帧的数量, 即seq_length
        mfcc = librosa.feature.mfcc(y=x, sr=sr, S=None, n_mfcc=20)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        # 光谱质心
        spectral_centroid = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
        # 光谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)[0]
        # 光谱斜率
        spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)[0]
        # 零交叉率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=x)[0]

        # 合并所有特征
        num_mfcc_coefficients = mfcc.shape[1]
        all_features = np.vstack([
            mfcc,
            delta_mfcc,
            delta2_mfcc,
            spectral_centroid.reshape(-1, num_mfcc_coefficients),  # 这里确保列数与mfcc相同
            spectral_bandwidth.reshape(-1, num_mfcc_coefficients),
            spectral_rolloff.reshape(-1, num_mfcc_coefficients),
            zero_crossing_rate.reshape(-1, num_mfcc_coefficients)  # 同样确保列数匹配
        ])

        all_features = all_features.T  # 返回形状为 (seq_length, features) 的特征矩阵

        # 将标签也转换为PyTorch张量
        label = torch.tensor(label)

        # 返回mfcc特征和标签
        return all_features, label

    def __len__(self):
        return len(self.labels)

#############################################
# 主程序入口点：测试自定义的数据集类SegDataset_mlp的功能
if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_1', type=str, default=r'drone_meta_small/drone_audio_annotations.csv',
                        help='root of data .csv')  # 存放数据集的位置，默认为'data/sudoku.csv'
    parser.add_argument('--data_root_2', type=str, default=r'drone_audio_small',
                        help='root of data .wave')  # 存放数据集的位置，默认为'data/sudoku.csv'
    opt = parser.parse_args()  # 解析命令行参数，将用户输入的参数值赋给opt变量


    # 测试read_file_list
    # train_paths, train_labels = read_file_list(root1=opt.data_root_1, root2=opt.data_root_2,type='train')
    # test_paths, test_labels = read_file_list(root1=opt.data_root_1, root2=opt.data_root_2,type='test')
    # print(type(train_paths))
    # print(train_paths[1], train_labels[1])

    # 测试traditional_seg()功能
    # features, labels = traditional_seg(root1=opt.data_root_1, root2=opt.data_root_2,type='train')
    # print(features[1].shape)
    # print('ok')



    # 测试SegDataset_mlp类
    # 实例化
    # voc_train = SegDataset_mlp(root1=opt.data_root_1, root2=opt.data_root_2,type='train')
    # # 打印数据集对象的类型，验证其为SegDataset_mlp的实例
    # print(type(voc_train))  # 预期输出：<class '__main__.SegDataset_mlp'>
    # # 打印数据集的长度，即数据集中样本的总数
    # print(len(voc_train))
    # # 通过索引访问数据集中第6个样本，获取其特征和标签
    # img, label = voc_train[6]
    # # 打印特征和标签的数据类型，验证数据处理正确性
    # print(type(img), type(label))
    # # 打印特征数据的形状，了解其维度信息
    # print(img.shape)
    # # 打印具体的标签值，确认数据加载无误
    # print(label)



    # 测试SegDataset_transformer类
    # 实例化
    voc_train = SegDataset_transformer(root1=opt.data_root_1, root2=opt.data_root_2,type='train')
    # 通过索引访问数据集中第6个样本，获取其特征和标签
    features, label = voc_train[6]
    # 打印特征和标签的数据类型，验证数据处理正确性
    print(type(features), type(label))
    # 打印特征数据的形状，了解其维度信息
    print(features.shape)
    # 打印具体的标签值，确认数据加载无误
    print(label)


