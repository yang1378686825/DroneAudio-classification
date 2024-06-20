import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class Transformer(nn.Module):
    def __init__(self, num_layers=6, d_model=20, nhead=4, dim_feedforward=128, dropout=0.1, n_categories=10):
        super(Transformer, self).__init__()

        # 定义超参数
        self.num_layers = num_layers  # 编码器层数量
        self.d_model = d_model   # 输入特征数
        self.nhead = nhead  # 自注意力头数
        self.dim_feedforward = dim_feedforward  # FFN中间层的维度
        self.dropout = dropout
        self.num_classes = n_categories

        # 创建Transformer编码器层
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                dim_feedforward=self.dim_feedforward,
                                                dropout=self.dropout)
        # 使用多个编码器层构建Transformer编码器
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # 输出分类头
        self.classification_head = nn.Linear(self.d_model, self.num_classes)

    def forward(self, src):
        # 输入数据维度: (batch_size, seq_length, features)
        transformer_output = self.transformer_encoder(src)  # (batch_size, seq_length, d_model)
        # 对序列的最后一个时间步的输出进行分类
        pooled_output = transformer_output[:, -1, :]  # 取序列的最后一帧
        output = self.classification_head(pooled_output)  # (batch_size, num_classes)
        return output


if __name__ == '__main__':

    # transformer模型测试_20个MFCC
    # 创建一个批次的随机数据，每个样本有32帧，每帧20个特征
    batch_size = 16
    seq_length = 32
    features = 20
    input_data = torch.randn(batch_size, seq_length, features)
    # 实例化模型
    model = Transformer(num_layers=6, d_model=features, nhead=int(features / 4), dim_feedforward=128, dropout=0.1, n_categories=5)
    # 将模型和数据转移到合适的设备（CPU或GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = input_data.to(device)
    # 通过模型运行前向传播
    outputs = model(input_data)
    # 打印输出的形状，应该为 (batch_size, num_classes)
    print("Output shape:", outputs.shape)
    # 打印一些输出值的例子
    print("Example outputs:\n", outputs[:2])  # 打印前两个样本的预测概率分布

    import librosa
    import numpy as np
