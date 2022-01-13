import d2l.torch
import torch.optim
from torch import nn
from matplotlib import pyplot as plt

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=(2, 2), )
        self.sigmoid_1 = nn.Sigmoid()
        self.pool_1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), )
        self.sigmoid_2 = nn.Sigmoid()
        self.pool_2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.flat_1 = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.sigmoid_3 = nn.Sigmoid()
        self.fc_2 = nn.Linear(120, 84)
        self.sigmoid_4 = nn.Sigmoid()
        self.fc_3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv_1(x)
        x = self.sigmoid_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.sigmoid_2(x)
        x = self.pool_2(x)
        x = self.flat_1(x)
        x = self.fc_1(x)
        x = self.sigmoid_3(x)
        x = self.fc_2(x)
        x = self.sigmoid_4(x)
        x = self.fc_3(x)
        return x


def evaluate_accuracy(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.torch.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.torch.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_net(net: nn.Module, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optim = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.torch.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    time, num_batches = d2l.torch.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.torch.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            time.start()
            optim.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optim.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.torch.accuracy(y_hat, y), X.shape[0])
            time.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))


        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / time.sum():.1f} examples/sec '
          f'on {str(device)}')
