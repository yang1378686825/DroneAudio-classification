import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import RNN, MLP
from segdataset import SegDataset_mlp, SegDataset_rnn
import argparse
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader,SubsetRandomSampler

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=r'data/sudoku.csv', help='root of data')
parser.add_argument('--load_root', type=str, default=r'checkpoints/mlp/ep007-loss0.805-val_loss0.787.pth', help='root of data')
# parser.add_argument('--load_root', type=str, default=r'checkpoints/rnn2/ep008-loss0.304-val_loss1.734.pth', help='root of data')
parser.add_argument('--epoch', type=int, default=10, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
parser.add_argument('--random_seed', type=int, default=10, help='random seed')
parser.add_argument('--model_kind', type=str, default='mlp', help='kind of model, i.e. CNN or RNN_LSTM or mlp or Transformer')
parser.add_argument('--is_shuffle_dataset', type=bool, default=True, help='shuffle dataset or not')
parser.add_argument('--test_split', type=float, default=0.2, help='ratio of the test set')
opt = parser.parse_args()

def main():
    if opt.model_kind == 'RNN_LSTM':
        model = RNN(,
        dataset = SegDataset_rnn(type='test')

    elif opt.model_kind == 'mlp':
        model = MLP()
        dataset = SegDataset_mlp(type='test')



    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sudoku_model = model.to(device)
    sudoku_model.load_state_dict(torch.load(opt.load_root))

    test_iter = torch.utils.data.DataLoader(dataset, batch_size=1,num_workers=opt.num_workers )

    if True:


        # test
        test_loss = 0
        correct=0
        total1=[]
        total2 = []
        sudoku_model.eval()
        with torch.no_grad():
            for index, (image, label) in enumerate(test_iter):
                image = image.to(device)
                label = label.to(device)

                output = sudoku_model(image)

                output=torch.argmax(output, dim=1)
                correct += (output == label).sum()
                output_numpy=output.detach().squeeze().cpu().numpy()
                label_numpy=label.detach().squeeze().cpu().numpy()
                total1.append(output_numpy)
                total2.append(label_numpy)
        accuracy = (correct / 80).item()
        print(accuracy)
        print(len(total1))
        print(total1)
        print(total2)
        print('准确率：', metrics.accuracy_score(total1, total2))
        confusion = confusion_matrix(total2, total1)
        print(confusion)
if __name__ == '__main__':
    main()