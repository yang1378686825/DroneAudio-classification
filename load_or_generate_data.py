import argparse
import os
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from segdataset_trad import traditional_seg, traditional_tree
from segdataset import SegDataset_rnn, SegDataset_mlp, SegDataset_transformer


def load_or_generate_data_SVM(filename, type='train', root1=r"meta/esc50.csv", root2=r"audio"):
    # 检查合并后的文件是否存在
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)      # 确保目录存在
        data_x, data_y = traditional_seg(root1=root1, root2=root2, type=type)       # 如果不存在，则生成数据
        combined_data = {'features': data_x, 'labels': data_y}  # 将数据打包到一个字典中

        # 将打包后的数据保存到pickle文件中
        with open(filename, 'wb') as handle:
            pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"数据已生成并保存至 {filename}")
        return combined_data['features'], combined_data['labels']

    else:
        # 如果文件存在，则从单一的pickle文件加载数据
        with open(filename, 'rb') as handle:
            combined_data = pickle.load(handle)

        print(f"数据已从 {filename} 加载")
        return combined_data['features'], combined_data['labels']



def load_or_generate_data_CLF(filename, type='train', root1=r"meta/esc50.csv", root2=r"audio"):
    # 检查合并后的文件是否存在
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)      # 确保目录存在
        data_x, data_y = traditional_tree(root1=root1, root2=root2, type=type)       # 如果不存在，则生成数据
        combined_data = {'features': data_x, 'labels': data_y}  # 将数据打包到一个字典中

        # 将打包后的数据保存到pickle文件中
        with open(filename, 'wb') as handle:
            pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"数据已生成并保存至 {filename}")
        return combined_data['features'], combined_data['labels']

    else:
        # 如果文件存在，则从单一的pickle文件加载数据
        with open(filename, 'rb') as handle:
            combined_data = pickle.load(handle)

        print(f"数据已从 {filename} 加载")
        return combined_data['features'], combined_data['labels']



def load_or_generate_data_RNN(filename, root1=r"meta/esc50.csv", root2=r"audio", type='train'):
    # 检查文件是否存在
    if not os.path.exists(filename) :
        # 不存在则生成并存储
        dataset = SegDataset_rnn(root1=root1, root2=root2, type=type)

        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Data generated and saved to {filename}")
        return dataset
    else:
        # 如果文件存在，则加载数据
        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle)
        print(f"Data loaded from {filename}")
        return dataset

def load_or_generate_data_Transformer(filename, root1=r"meta/esc50.csv", root2=r"audio", type='train'):
    # 检查文件是否存在
    if not os.path.exists(filename) :
        # 不存在则生成并存储
        dataset = SegDataset_transformer(root1=root1, root2=root2, type=type)

        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Data generated and saved to {filename}")
        return dataset
    else:
        # 如果文件存在，则加载数据
        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle)
        print(f"Data loaded from {filename}")
        return dataset

def load_or_generate_data_MLP(filename, root1=r"meta/esc50.csv", root2=r"audio", type='train'):
    # 检查文件是否存在
    if not os.path.exists(filename) :
        # 不存在则生成并存储
        dataset = SegDataset_mlp(root1=root1, root2=root2, type=type)

        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Data generated and saved to {filename}")
        return dataset
    else:
        # 如果文件存在，则加载数据
        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle)
        print(f"Data loaded from {filename}")
        return dataset


if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_1', type=str, default=r'drone_meta_small/drone_audio_annotations.csv',
                        help='root of data .csv')  # 存放数据集的位置，默认为'data/sudoku.csv'
    parser.add_argument('--data_root_2', type=str, default=r'drone_audio_small',
                        help='root of data .wave')  # 存放数据集的位置，默认为'data/sudoku.csv'
    opt = parser.parse_args()  # 解析命令行参数，将用户输入的参数值赋给opt变量

    # SVM的dataset测试
    # print('SVM的dataset测试')
    # dataset_x, dataset_y = load_or_generate_data_SVM('SVM_features/SVM_features_train.pickle', root1=opt.data_root_1, root2=opt.data_root_2,type='train')
    # # datasetx, datasety = traditional_seg(type='train')
    # #
    # # if dataset_y == datasety:
    # #     print("dataset_ 和 dataset 相同")
    # # else:
    # #     print("dataset_ 和 dataset 不同")
    #
    # features = dataset_x[6]
    # label = dataset_y[6]
    # print(len(features))



    # CLF的dataset测试
    # print('CLF的dataset测试')
    # dataset_x, dataset_y = load_or_generate_data_CLF('CLF_features/CLF_features_train.pickle', root1=opt.data_root_1, root2=opt.data_root_2,type='train')
    # # datasetx, datasety = traditional_seg(type='train')
    # #
    # # if dataset_y == datasety:
    # #     print("dataset_ 和 dataset 相同")
    # # else:
    # #     print("dataset_ 和 dataset 不同")
    #
    # features = dataset_x[6]
    # label = dataset_y[6]
    # print(len(features))



    # RNN的dataset测试
    print('RNN的dataset测试')
    dataset_ = load_or_generate_data_RNN('RNN_features/RNN_features_train.pickle', root1=opt.data_root_1, root2=opt.data_root_2,type='train')
    dataset_1 = load_or_generate_data_RNN('RNN_features/RNN_features_test.pickle', root1=opt.data_root_1, root2=opt.data_root_2,type='test')

    dataset = SegDataset_rnn(root1=opt.data_root_1, root2=opt.data_root_2,)
    dataset1 = SegDataset_rnn(root1=opt.data_root_1, root2=opt.data_root_2, type='test')

    if dataset_.labels == dataset.labels:
        print("dataset_ 和 dataset 相同")
    else:
        print("dataset_ 和 dataset 不同")

    features, label = dataset[6]
    print(features.shape)
    print(features.shape[0], features.shape[1])  #sr = 44100 :87, 20  # sr = 16100 :32, 20



    # MLP的dataset测试
    # print('MLP的dataset测试')
    # dataset_ = load_or_generate_data_MLP('MLP_features/MLP_features_train.pickle', root1=opt.data_root_1,
    #                                      root2=opt.data_root_2, type='train')
    # dataset_1 = load_or_generate_data_MLP('MLP_features/MLP_features_test.pickle', root1=opt.data_root_1,
    #                                       root2=opt.data_root_2, type='test')
    #
    # dataset = SegDataset_mlp(root1=opt.data_root_1, root2=opt.data_root_2, )
    # dataset1 = SegDataset_mlp(root1=opt.data_root_1, root2=opt.data_root_2, type='test')
    #
    # if dataset_.labels == dataset.labels:
    #     print("dataset_ 和 dataset 相同")
    # else:
    #     print("dataset_ 和 dataset 不同")
    #
    # features, label = dataset[6]
    # print(features.shape)  # torch.Size([51])

