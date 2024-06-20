import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

from load_or_generate_data import load_or_generate_data_RNN, load_or_generate_data_MLP, \
    load_or_generate_data_Transformer
from model_CNN import CNN, CRNN_LSTM
from model_Transformer import Transformer
from models import RNN, MLP
from segdataset import SegDataset_mlp,SegDataset_rnn
import argparse
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_1', type=str, default=r'drone_meta_small/drone_audio_annotations.csv', help='root of data .csv')  # 存放数据集的位置，默认为'data/sudoku.csv'
parser.add_argument('--data_root_2', type=str, default=r'drone_audio_small', help='root of data .wave')  # 存放数据集的位置，默认为'data/sudoku.csv'
# parser.add_argument('--save_root', type=str, default=r'checkpoints/RNN_LSTM', help='root of data')  # 模型保存的根目录，默认为'checkpoints/mlp'
parser.add_argument('--epoch', type=int, default=6, help='epoch number')  # 训练轮数，默认为0~epoch-1
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  # 学习率，默认为1e-3
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')  # 批次大小，默认为16
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')  # 数据加载时使用的子进程数，默认为8
parser.add_argument('--random_seed', type=int, default=10, help='random seed')  # 随机种子，用于确保实验的可复现性，默认为10
parser.add_argument('--is_shuffle_dataset', type=bool, default=True, help='shuffle dataset or not')  # 是否打乱数据集，默认为True
parser.add_argument('--test_split', type=float, default=0.1, help='ratio of the test set')  # 测试集划分比例，默认为0.1

parser.add_argument('--model_kind', type=str, default='Transformer', help='kind of model, i.e. CNN or RNN_LSTM or mlp or Transformer')  # 模型类型，可以选择‘CNN’、‘RNN_LSTM’、'CRNN_LSTM’、'Transformer'、‘mlp’、
opt = parser.parse_args()  # 解析命令行参数，将用户输入的参数值赋给opt变量

def main():
    # 识别分类类别数目
    df = pd.read_csv(opt.data_root_1)
    category_counts = df['category'].nunique()+1
    print(f"类别数目: {category_counts}")

    if opt.model_kind == 'RNN_LSTM':
        dataset = load_or_generate_data_RNN(filename='RNN_features/RNN_features_train.pickle', root1=opt.data_root_1, root2=opt.data_root_2, type='train')
        dataset1 = load_or_generate_data_RNN(filename='RNN_features/RNN_features_test.pickle', root1=opt.data_root_1, root2=opt.data_root_2, type='test')
        features, _ = dataset[6]  # 获取其中一个样本的特征向量，将其维度传入model，以便于构建合适的输入维度
        model = RNN(n_features=features.shape[1], seq_length=features.shape[0], n_categories=category_counts)
    elif opt.model_kind == 'mlp':
        dataset = load_or_generate_data_MLP(filename='MLP_features/MLP_features_train.pickle', root1=opt.data_root_1, root2=opt.data_root_2, type='train')
        dataset1 = load_or_generate_data_MLP(filename='MLP_features/MLP_features_test.pickle', root1=opt.data_root_1, root2=opt.data_root_2, type='test')
        features, _ = dataset[6]
        model = MLP(n_features=len(features), n_categories=category_counts)
    elif opt.model_kind == 'CNN':  # CNN网络使用和RNN相同的输入特征
        dataset = load_or_generate_data_RNN(filename='RNN_features/RNN_features_train.pickle', root1=opt.data_root_1, root2=opt.data_root_2, type='train')
        dataset1 = load_or_generate_data_RNN(filename='RNN_features/RNN_features_test.pickle', root1=opt.data_root_1, root2=opt.data_root_2, type='test')
        features, _ = dataset[6]  # 获取其中一个样本的特征向量，将其维度传入model，以便于构建合适的输入维度
        model = CNN(n_features=features.shape[1], seq_length=features.shape[0], n_categories=category_counts)
    elif opt.model_kind == 'CRNN_LSTM':  # CRNN网络使用和RNN相同的输入特征
        dataset = load_or_generate_data_RNN(filename='RNN_features/RNN_features_train.pickle', root1=opt.data_root_1, root2=opt.data_root_2, type='train')
        dataset1 = load_or_generate_data_RNN(filename='RNN_features/RNN_features_test.pickle', root1=opt.data_root_1, root2=opt.data_root_2, type='test')
        features, _ = dataset[6]  # 获取其中一个样本的特征向量，将其维度传入model，以便于构建合适的输入维度
        model = CRNN_LSTM(n_features=features.shape[1], seq_length=features.shape[0], n_categories=category_counts)
    elif opt.model_kind == 'Transformer':  # Transformer
        dataset = load_or_generate_data_Transformer(filename='Transformer_features/Transformer_features_train.pickle', root1=opt.data_root_1, root2=opt.data_root_2, type='train')
        dataset1 = load_or_generate_data_Transformer(filename='Transformer_features/Transformer_features_test.pickle', root1=opt.data_root_1, root2=opt.data_root_2, type='test')
        features, _ = dataset[6]  # 获取其中一个样本的特征向量，将其维度传入model，以便于构建合适的输入维度
        model = Transformer(d_model=features.shape[1], nhead=int(features.shape[1] / 8), n_categories=category_counts)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"{opt.model_kind} training at {device} for {opt.epoch} epochs")

    all_train_epoch_loss = []
    all_test_epoch_loss = []
    all_test_epoch_accuracy=[]
    sudoku_model = model.to(device)


    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(sudoku_model.parameters(), opt.lr)
    # 定义指数衰减策略
    # 每个iteration后，学习率乘以gamma
    scheduler = ExponentialLR(optimizer, gamma=0.997)  # 这里假设每迭代一次，学习率衰减为原来的0.997倍


    train_iter = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,num_workers=opt.num_workers)
    test_iter = torch.utils.data.DataLoader(dataset1, batch_size=opt.batch_size,num_workers=opt.num_workers)

    save_root = 'checkpoints'
    save_root = os.path.join(save_root, opt.model_kind)
    # 检查该目录是否存在，不存在则创建
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for epo in range(opt.epoch):
        print('<---------------------------------------------------->')
        print('epoch: %f' % epo)
        train_loss = 0
        sudoku_model.train()  # 启用batch normalization和drop out
        for index, (features, label) in enumerate(train_iter):
            features = features.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = sudoku_model(features)
            loss = criterion(output, label)
            loss.backward()
            iter_loss = loss.item()
            train_loss += iter_loss
            optimizer.step()
            # 每个iteration后调整学习率
            scheduler.step()  # 不需要传递epoch参数

            if np.mod(index, 5) == 0 or index == len(train_iter) - 1: # 添加检查是否为最后一个迭代
                current_lr = scheduler.get_last_lr()[0]  # 获取当前的学习率
                print('epoch {}, {}/{}batches,train loss is {}, lr: {}'.format(epo, index, len(train_iter), iter_loss, current_lr))

        # test
        test_loss = 0
        correct=0
        total=0
        sudoku_model.eval()
        with torch.no_grad():
            for index, (features, label) in enumerate(test_iter):
                features = features.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = sudoku_model(features)
                loss = criterion(output, label)
                output=torch.argmax(output, dim=1)
                correct += (output == label).sum()
                total += len(label.view(-1))
                iter_loss = loss.item()
                test_loss += iter_loss
        accuracy=(correct/total).item()
        print('epoch train loss = %f, epoch test loss = %f,accuracy =%.3f'
              % (train_loss / len(train_iter), test_loss / len(test_iter),accuracy))

        if np.mod(epo, 1) == 0:
            # 只存储模型参数
            torch.save(sudoku_model.state_dict(), save_root+'/ep%03d-loss%.3f-val_loss%.3f.pth' % (
                epo, (train_loss / len(train_iter)), (test_loss / len(test_iter)))
                       )
            print('saving checkpoints/model_{}.pth'.format(epo))

        all_test_epoch_accuracy.append(accuracy)
        all_train_epoch_loss.append(train_loss / len(train_iter))
        all_test_epoch_loss.append(test_loss / len(test_iter))



    # 创建图表
    fig1, ax1 = plt.subplots()
    ax1.set_title('train_loss')
    ax1.plot(all_train_epoch_loss)
    ax1.set_xlabel('epoch')

    fig2, ax2 = plt.subplots()
    ax2.set_title('test_loss')
    ax2.plot(all_test_epoch_loss)
    ax2.set_xlabel('epoch')

    fig3, ax3 = plt.subplots()
    ax3.set_title('test_accuracy')
    ax3.plot(all_test_epoch_accuracy)
    ax3.set_xlabel('epoch')

    # 保存图表
    plt.ioff()  # 关闭交互模式，这对于非GUI环境（如后台脚本）是必要的
    try:
        fig1.savefig(os.path.join(save_root, 'train_loss.png'))
        fig2.savefig(os.path.join(save_root, 'test_loss.png'))
        fig3.savefig(os.path.join(save_root, 'test_accuracy.png'))
    finally:
        plt.close('all')  # 关闭所有图表，释放资源
if __name__ == '__main__':
    main()