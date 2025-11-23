import torch
from torch import nn
from d2l import torch as d2l
import my_utils
import matplotlib.pyplot as plt

# 1. 定义不需要依赖全局变量的基础函数（如激活函数）
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# ==========================================
# 关键修改：将所有执行逻辑放入 if __name__ == '__main__':
# ==========================================
if __name__ == '__main__':
    # 设置 Matplotlib 后端 (可选)
    # plt.switch_backend('TkAgg')

    batch_size = 256
    # load_data_fashion_mnist 在 Windows 下默认可能会开启多进程
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 定义输入输出隐藏层数量
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    # 初始化权重和偏置
    # 注意：变量定义在 if 块内，这样子进程 import 模块时不会重复初始化这些变量
    W1 = nn.Parameter(torch.randn(
        num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(
        num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    # 实现网络模型
    # 注意：因为 net 依赖 W1, b1 等变量，所以 net 也必须定义在这些变量之后（即 if 块内）
    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)
        return (H @ W2 + b2)

    # 定义交叉熵损失函数
    loss = nn.CrossEntropyLoss(reduction='none')

    # 定义训练参数
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)

    print("开始训练...")
    # 这一行是报错的根源，现在被保护在 if __name__ == '__main__' 下了
    my_utils.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

    print("训练结束，进行预测...")
    d2l.predict_ch3(net, test_iter)

    print("正在显示图像...")
    plt.show()