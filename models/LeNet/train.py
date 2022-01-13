from d2l import torch as d2l
import torch
from LeNet import *
from matplotlib import pyplot as plt

if __name__ == '__main__':
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    LeNet = LeNet()
    batch_size = 256
    train_iter, test_iter = d2l.torch.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.3, 10
    train_net(LeNet, train_iter, test_iter, num_epochs, lr, d2l.torch.try_gpu())
    plt.show()

