import torch
from AlexNet import *
from matplotlib import pyplot as plt

if __name__ == '__main__':
    X = torch.randn(1, 1, 224, 224)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.01, 10
    train_net(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    plt.show()
