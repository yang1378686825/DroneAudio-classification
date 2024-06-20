import torch
import torch.nn as nn
import torchvision


# 定义一个自定义的Reshape层，用于在神经网络中动态改变数据张量的形状
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()   # 调用父类nn.Module的初始化方法，确保必要的初始化过程
        self.shape = args  # 存储需要编程的的形状参数，用于后续重塑张量

    def forward(self, x):
        # 使用view方法重塑张量x的形状
        # x.size(0) ：输入张量的第一个维度（通常是批量大小），其保持不变
        # +self.shape 将初始化时传入的形状参数附加在第一个维度之后
        return x.view((x.size(0),) + self.shape)



class MLP(nn.Module):
    def __init__(self, n_features, dropout_rate=0.2, n_categories=10):  # 添加dropout_rate参数，默认值为0.2
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)  # 使用传入的dropout_rate
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.out = nn.Linear(128, n_categories)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x



class RNN_LSTM(nn.Module):
    # 定义一个循环神经网络（RNN）模块，基于LSTM单元，继承自nn.Module
    def __init__(self, n_features, seq_length, n_categories=10):
        super().__init__()  # 继承nn.Module的初始化方法

        # LSTM层定义
        self.LSTM = nn.LSTM(input_size=n_features, hidden_size=32, batch_first=True, num_layers=2)
        # input_size: 输入特征的维度，这里是20(即20个MFCC特征)
        # hidden_size: LSTM隐藏状态的维度，这里是32
        # batch_first: 设为True表示输入数据的形状是(batch_size, sequence_length, input_size)，便于处理批量数据
        # num_layers: LSTM的层数，这里是2层

        self.flatten = nn.Flatten()   # Flatten层，用于将LSTM的输出展平，以便于送入全连接层

        # 第一个全连接层，将LSTM展平后的输出映射到4096维
        self.linear1 = nn.Linear(32 * seq_length, 4096)
        # 注意：这里的32*431假设是LSTM输出的形状在展平后的尺寸，其中seq_length=431代表序列长度
        # 实际应用中需根据输入序列的真实最大长度调整此值。

        # 第二个全连接层，将4096维映射到最终的输出维度，这里为10（例如，进行10分类任务）
        self.linear2 = nn.Linear(4096, n_categories)

        # ReLU激活函数，用于增加网络的非线性
        self.relu = nn.ReLU()

        # Dropout层，用于防止过拟合，此处未指定dropout概率，默认为0.5
        self.dropout = nn.Dropout()

    def forward(self, x):
        # 前向传播函数，定义了数据通过网络的流程
        # 通过LSTM处理输入x，返回全部时间步的输出和隐藏状态，我们仅取输出部分([0])
        x = self.LSTM(x)[0]

        # 将LSTM的输出展平，准备送入全连接层
        x = self.flatten(x)

        # 第一个全连接层及激活函数
        x = self.linear1(x)
        x = self.relu(x)

        # 应用Dropout
        x = self.dropout(x)

        # 第二个全连接层，产生最终输出
        x = self.linear2(x)

        return x  # 返回模型的预测输出


if __name__ == '__main__':

    # 示例使用Reshape类
    # 假设我们有一个形状为(2, 3, 4, 5)的张量，想要调整形状为(2, 4, 5, 3)
    reshape_layer = Reshape(3, 5, 4)
    input_tensor = torch.randn(2, 4, 5, 3)  # 随机生成一个示例输入张量
    output_tensor = reshape_layer(input_tensor)
    print(output_tensor.shape)  # 应输出torch.Size([2, 4, 5, 3])


    # 示例测试MLP类
    # 创建一个默认dropout率的模型
    model_default = MLP()
    # 创建一个dropout率为0.5的模型
    model_custom_dropout = MLP(dropout_rate=0.5)
    # 创建一个随机输入张量，模拟输入数据
    input_data = torch.randn(10, 45)  # 假设一批有10个样本，每个样本45维
    # 通过模型进行前向传播
    try:
        output_default_dropout = model_default(input_data)
        output_custom_dropout = model_custom_dropout(input_data)
        print("Model output shape:", output_default_dropout.shape, output_custom_dropout.shape)  # 应该输出(10, 10)，表示10个样本，每个样本10维的预测概率
    except Exception as e:
        print("An error occurred:", e)


    # 示例测试RNN
    # 确保使用的是CUDA，如果可用的话
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # 实例化RNN模型
    rnn_model = RNN_LSTM().to(device)
    # 假设输入数据是一批序列数据，每个序列长度为431，每个时间步有20个特征
    batch_size = 16  # 批次大小
    sequence_length = 431  # 序列长度
    input_features = 20  # 每个时间步的特征数
    # 创建模拟输入数据
    input_data = torch.randn(batch_size, sequence_length, input_features).to(device)
    # 前向传播
    output = rnn_model(input_data)
    # 打印输出的形状，以确认网络工作正常
    print("Output shape:", output.shape)  # # 应该输出(16, 10)，表示16个样本，每个样本10维(十类)的预测概率
