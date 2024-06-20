import argparse
import librosa
import numpy as np
import torch
from torch import nn
from segdataset import read_file_list
'''
为什么是以为数据：
features=132是通道数
seq_length=32是数据维度，即一维数据，用conv1d
RGB图像则是3通道的二维数据
'''

class SegDataset_MelSpecCNN(torch.utils.data.Dataset):
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

        y, sr = librosa.load(signal, sr=16100)

        # 计算梅尔频谱图
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        # 光谱质心
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        # 光谱带宽
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        # 光谱斜率
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        # 零交叉率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]

        # 合并所有特征
        num_S_db_coefficients = S_db.shape[1]
        all_features = np.vstack([
            S_db,
            spectral_centroid.reshape(-1, num_S_db_coefficients),  # 这里确保列数与mfcc相同
            spectral_bandwidth.reshape(-1, num_S_db_coefficients),
            spectral_rolloff.reshape(-1, num_S_db_coefficients),
            zero_crossing_rate.reshape(-1, num_S_db_coefficients)  # 同样确保列数匹配
        ])

        all_features = all_features.T  # 返回形状为 (seq_length, features) 的特征矩阵

        # 将标签也转换为PyTorch张量
        label = torch.tensor(label)

        # 返回mfcc特征和标签
        return all_features, label

    def __len__(self):
        return len(self.labels)


class MelSpecCNN(nn.Module):
    def __init__(self, n_features=132, seq_length=32, dropout_rate=0.2, n_categories=5):
        super().__init__()

        # 输入通道数对应于梅尔频谱的特征数量
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # 添加了BatchNorm以提升训练稳定性
            nn.Dropout(p=dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate)
        )

        # 计算全连接层的输入维度。假设seq_length在卷积后保持不变，通道数变为256
        self.fc = nn.Linear(256 * seq_length, n_categories)

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)  # 转换为(batch_size, features, seq_length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 展平特征
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


# 定义一维残差块，核心组件之一，用于构建ResNet模型
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.2):
        super(ResidualBlock1D, self).__init__()

        # 第一层卷积，增加通道数并学习特征
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # 批量归一化，加速训练并提高模型的泛化性能
        self.bn1 = nn.BatchNorm1d(out_channels)
        # ReLU激活函数，增加非线性
        self.relu = nn.ReLU(inplace=True)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(p=dropout_rate)
        # 第二层卷积，进一步提炼特征
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果原本x的通道数(即in_channels)和经过残差块输出的通道数(即out_channels)不同，需要调整shortcut连接
        # 使用1x1卷积调整通道数，保持尺寸一致
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1,
                                                                                    stride)

    def forward(self, x):
        # 残差连接的输入分支
        residual = x
        # 第一层卷积及后续处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # 第二层卷积及后续处理
        x = self.conv2(x)
        x = self.bn2(x)

        # 加上快捷连接，保持特征的直接传递路径
        x += self.shortcut(residual)
        # 再次激活，引入非线性
        x = self.relu(x)

        return x


# 一维ResNet模型类
class ResNetLike1D(nn.Module):
    def __init__(self, n_features=132, seq_length=32, n_categories=10, base_filters=32, num_blocks=2,
                 dropout_rate=0.2):
        super(ResNetLike1D, self).__init__()

        # 初始卷积层，减少特征尺寸并提取初步特征
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=base_filters, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层，进一步降低尺寸，减少计算复杂度
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 定义残差块的堆叠，根据num_blocks生成不同数量的残差块
        filters = [base_filters] * num_blocks  # 每个block的输出通道数保持一致或递增，此处简化处理为一致
        # 使用ModuleList动态管理残差块，便于后续遍历
        self.blocks = nn.ModuleList([
            # 如果不是最后一个block，输出通道数翻倍；如果是最后一个，保持不变
            ResidualBlock1D(filters[i], filters[i + 1] if i < len(filters) - 1 else filters[i],
                            dropout_rate=dropout_rate)
            for i in range(len(filters) - 1)
        ])

        # 平均池化层，用于在最后一个残差块后压缩特征图的时间维度
        # 计算合适的池化尺寸，保证输出可以被全连接层接收
        self.avgpool = nn.AvgPool1d(seq_length // (2 ** len(filters)))
        # 全连接层，进行分类
        self.fc = nn.Linear(filters[-1], n_categories)

    def forward(self, x):
        # 确保输入数据为float类型，兼容模型运算
        x = x.float()
        # (batch_size, seq_length, features) -> (batch_size, features, seq_length)这是卷积神经网络对一维序列数据的标准输入格式。
        x = x.permute(0, 2, 1)

        # 初始卷积和池化处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 依次通过残差块
        for block in self.blocks:
            x = block(x)

        # 平均池化和展平特征，准备进入全连接层
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # 分类预测
        x = self.fc(x)

        return x



if __name__ == '__main__':

    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_1', type=str, default=r'drone_meta_small/drone_audio_annotations.csv',
                        help='root of data .csv')  # 存放数据集的位置，默认为'data/sudoku.csv'
    parser.add_argument('--data_root_2', type=str, default=r'drone_audio_small',
                        help='root of data .wave')  # 存放数据集的位置，默认为'data/sudoku.csv'
    opt = parser.parse_args()  # 解析命令行参数，将用户输入的参数值赋给opt变量


    # 测试SegDataset_MelSpecCNN类
    # 实例化
    voc_train = SegDataset_MelSpecCNN(root1=opt.data_root_1, root2=opt.data_root_2,type='train')
    # 通过索引访问数据集中第6个样本，获取其特征和标签
    features, label = voc_train[6]
    # 打印特征和标签的数据类型，验证数据处理正确性
    print(type(features), type(label))
    # 打印特征数据的形状，了解其维度信息
    print(features.shape)
    # 打印具体的标签值，确认数据加载无误
    print(label)



    # MelSpecCNN/ResNetLike1D模型测试132个features
    train_iter = torch.utils.data.DataLoader(voc_train, batch_size=16)
    # 实例化模型
    # model = MelSpecCNN(n_categories=5)
    model = ResNetLike1D(n_categories=5)
    # 将模型和数据转移到合适的设备（CPU或GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 获取第一个批次的数据,是一个包含样本和标签的元组，如(batch_data, batch_labels)
    first_batch = next(iter(train_iter))
    batch_data = first_batch[0].to(device)
    batch_labels = first_batch[1].to(device)


    # 通过模型运行前向传播
    outputs = model(batch_data)
    # 打印输出的形状，应该为 (batch_size, num_classes)
    print("Output shape:", outputs.shape)
    # 打印一些输出值的例子
    print("Example outputs:\n", outputs[:2])  # 打印前两个样本的预测概率分布