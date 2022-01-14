import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 100
    data_path = '/Users/william/Desktop/project/pyspace/study/data'

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_data_iter = iter(test_loader)
    val_image, val_label = val_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))




    # val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=False, transform=transform)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
    #                                          shuffle=False, num_workers=0)

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 50 == 49:  # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    train()
