# ESC-10数据集的音频分类

本文选取了ESC-10数据集作为音频分类的数据集，选取了四种方法进行分类，包括支持向量机(SVM)，决策树(CLF)，多层感知机(MLP)，循环神经网络(RNN)。最终支持向量机取得了最优的结果（0.825）。
但实际上是从esc-50中筛出来带有esc-10标签的数据，变成和原本的esc-10相同的400条数据，十个类别。这十个类别的target是：
# ESC-10特定标签映射（示例中未完整列出所有映射）
    label_ESC10 = [0, 1, 10, 11, 12, 20, 21, 38, 40, 41]

### 多层感知机（MLP）

学习率设置为0.001，优化算法选择Adam，损失函数为交叉熵函数，训练10个epoch的损失函数变化如下所示：

训练集上的损失函数:

![](https://github.com/deepxzy/ESC-50-Audio-classification/blob/master/checkpoints/mlp/train_loss.png)

测试集上的损失函数:

![](https://github.com/deepxzy/ESC-50-Audio-classification/blob/master/checkpoints/mlp/test_loss.png)

测试集上的准确率:

![](https://github.com/deepxzy/ESC-50-Audio-classification/blob/master/checkpoints/mlp/test_accury.png)



### 循环神经网络（RNN）

训练集上的损失函数:

![](https://github.com/deepxzy/ESC-50-Audio-classification/blob/master/checkpoints/rnn2/train_loss.png)

测试集上的损失函数:

![](https://github.com/deepxzy/ESC-50-Audio-classification/blob/master/checkpoints/rnn2/test_loss.png)

测试集上的准确率:

![](https://github.com/deepxzy/ESC-50-Audio-classification/blob/master/checkpoints/rnn2/test_accury.png)

## 总结

|          |                   支持向量机(SVM)                    |                     决策树(CLF)                      |                   多层感知机(MLP)                    | 循环神经网络(RNN) |
| :------: | :---------------------------------------------: | :---------------------------------------------: | :---------------------------------------------: | :----------: |
|   特征   | mfcc，chroma，melspectrogram，contrast，tonnetz | mfcc，chroma，melspectrogram，contrast，tonnetz | mfcc，chroma，melspectrogram，contrast，tonnetz |     mfcc     |
| 特征维度 |                    （1，45）                    |                    （1，5）                     |                    （1，45）                    | （431, 20）  |
|  准确率  |                      0.825                      |                      0.650                      |                      0.762                      |    0.750     |

