import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

#线性输入输出数据集的生成
def synthetic_data(w, b, num_examples): 
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b#matmul为数学意义上的矩阵乘法
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))#-1表示自动推算维度大小（此处为num_examples），最终y的形状变为(num_examples, 1)

#能打乱数据集中的样本并以小批量方式获取数据。

#在下面的代码中，我们定义一个data_iter函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。 每个小批量包含一组特征和标签。
def data_iter(batch_size, features, labels):
    num_exfor epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')amples = len(features)
    indices = list(range(num_examples))#生成长度为num_examples的标签列表
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)#将标签数组打乱
    #生成每个batch的标签
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])#取标签数组的一个切片，大小为batch_size（左闭右开）
        yield features[batch_indices], labels[batch_indices]#执行到yield函数暂停data_iter,"下次迭代该函数返回的生成器对象"继续执行该函数


#定义线性回归模型
def linreg(X, w, b):
    """线性回归"""
    return torch.matmul(X, w) + b

#定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 #使用reshape强制二者形状一致

#定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():# 临时禁用 PyTorch 的梯度追踪机制，确保参数更新（param -= ...）和梯度清零（param.grad.zero_()）
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()





true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

#为生成的y和x作图
plt.figure(figsize=(6, 4))  
plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)

#初始化模型参数，权重w和偏置b
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)#二行一列，样本有两个特征和一个预测值
b = torch.zeros(1, requires_grad=True)#长度为一的一维张量

#训练
batch_size = 10
lr = 0.03
num_epochs = 3
net = linreg#线性回归模型即为一个简单的线性神经网络
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()  #对loss先求和再进行反向传播求导，可以获得小批量总loss对于w，b的导数
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
