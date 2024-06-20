import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, n_features=20, seq_length=32, dropout_rate=0.2, n_categories=10):
        super().__init__()

        # 这里n_features是输入通道数，seq_length是序列的长度（宽度）
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # 计算全连接层的输入维度。假设seq_length在卷积后保持不变（未使用池化层减少序列长度），通道数变为128
        # 注意：实际应用中，若使用了池化层导致序列长度变化，则需相应调整
        self.fc = nn.Linear(128 * seq_length, n_categories)

    def forward(self, x):
        # 将输入从(batch_size, seq_length, n_features)调整为(batch_size, n_features, seq_length)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 展平特征
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x



class CRNN_LSTM(nn.Module):
    def __init__(self, n_features=20, seq_length=32, n_categories=10, dropout_rate=0.2):
        super(CRNN_LSTM, self).__init__()

        # CNN部分用于提取空间特征
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 添加池化层以减小序列长度
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 可选的第二个池化层
        )
        # # 计算经过CNN处理后的序列长度
        # reduced_seq_length = seq_length // (2 * 2) if seq_length > 4 else 1  # 假设使用了两次池化，每次减半

        # LSTM部分用于捕捉时间序列特征
        self.rnn = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, num_layers=2, dropout=dropout_rate)

        # 全连接层用于最终分类
        self.fc = nn.Linear(512, n_categories)  # 使用隐藏状态作为全连接层的输入

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 适应CNN的输入
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # 重新调整形状以适应RNN

        # 通过LSTM处理
        x, (hn, _) = self.rnn(x)

        # 单向LSTM，收集两个layer的隐藏状态并拼接
        all_layers_hidden = torch.cat([layer_state for layer_state in hn], dim=1)

        x = self.fc(all_layers_hidden)

        return x



if __name__ == '__main__':
    # 测试RNN
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
    # cnn_model = CNN(n_features=input_features, seq_length=sequence_length).to(device)
    # # 前向传播
    # output = cnn_model(input_data)
    # # 打印输出的形状，以确认网络工作正常
    # print("Output shape:", output.shape)  # # 应该输出(16, 10)，表示16个样本，每个样本10维(十类)的预测概率


    # 示例测试CRNN
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
    cnn_model = CRNN_LSTM(n_features=input_features, seq_length=sequence_length).to(device)
    # 前向传播
    output = cnn_model(input_data)
    # 打印输出的形状，以确认网络工作正常
    print("Output shape:", output.shape)  # # 应该输出(16, 10)，表示16个样本，每个样本10维(十类)的预测概率