import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

#线性输入输出数据的生成
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b#matmul为数学意义上的矩阵乘法
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))#-1表示自动推算维度大小（此处为num_examples），最终y的形状变为(num_examples, 1)

#
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

#为生成的y和x作图
plt.figure(figsize=(6, 4))
plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)

