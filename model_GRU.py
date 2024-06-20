import torch
import torch.nn as nn
'''在GRU的情况下，我们直接使用最后一个时间步的输出送入全连接层，因为GRU没有像LSTM那样的细胞状态和隐藏状态分离。'''

class RNN_GRU(nn.Module):
    def __init__(self, n_features=64, hidden_size=128, n_categories=10):
        super(RNN_GRU, self).__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_categories)

    def forward(self, x):
        out, _ = self.gru(x)  # 输出和隐藏状态，我们只关心输出
        out = out[:, -1, :]  # 取序列的最后一时刻输出作为全连接层的输入
        out = self.fc(out)
        return torch.softmax(out, dim=-1)


class CRNN_GRU(nn.Module):
    def __init__(self, n_features=20, seq_length=32, n_categories=10, dropout_rate=0.2):
        super(CRNN_GRU, self).__init__()

        # CNN部分保持不变，用于提取空间特征
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # GRU部分替换LSTM，用于捕捉时间序列特征
        self.rnn = nn.GRU(input_size=128, hidden_size=256, batch_first=True, num_layers=2, dropout=dropout_rate)

        # 全连接层同样保持不变，用于最终分类
        self.fc = nn.Linear(256, n_categories)  # GRU的隐藏层大小直接作为全连接层的输入

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整形状以适应CNN的输入
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # 处理后再次调整形状以适应GRU

        # 通过GRU处理
        x, _ = self.rnn(x)  # GRU的输出和隐藏状态合并为一个张量，因此我们只需要x

        # 使用最后一个时间步的输出进行分类
        x = x[:, -1, :]  # 选取最后一个时间步的输出

        x = self.fc(x)

        return x


if __name__ == '__main__':
    # 示例测试CRNN_GRU
    # 确保使用的是CUDA，如果可用的话
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # 假设输入数据是一批序列数据，每个序列长度为32，每个时间步有20个特征
    batch_size = 16  # 批次大小
    sequence_length = 32  # 序列长度
    input_features = 20  # 每个时间步的特征数
    # 创建模拟输入数据
    input_data = torch.randn(batch_size, sequence_length, input_features).to(device)
    # 实例化CRNN模型
    cnn_model = CRNN_GRU(n_features=input_features, seq_length=sequence_length).to(device)
    # 前向传播
    output = cnn_model(input_data)
    # 打印输出的形状，以确认网络工作正常
    print("Output shape:", output.shape)  # # 应该输出(16, 10)，表示16个样本，每个样本10维(十类)的预测概率

    # 示例测试GRU
    # # 确保使用的是CUDA，如果可用的话
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device:", device)
    # # 假设输入数据是一批序列数据，每个序列长度为32，每个时间步有20个特征
    # batch_size = 16  # 批次大小
    # sequence_length = 32  # 序列长度
    # input_features = 20  # 每个时间步的特征数
    # # 创建模拟输入数据
    # input_data = torch.randn(batch_size, sequence_length, input_features).to(device)
    # # 实例化CNN模型
    # cnn_model = RNN_GRU(n_features=input_features).to(device)
    # # 前向传播
    # output = cnn_model(input_data)
    # # 打印输出的形状，以确认网络工作正常
    # print("Output shape:", output.shape)  # # 应该输出(16, 10)，表示16个样本，每个样本10维(十类)的预测概率
