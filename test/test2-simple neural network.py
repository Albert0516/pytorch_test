import torch
import torch.nn as nn
import torch.nn.functional as F


# test for neural network


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()
        # 卷积层 ‘1’ 表示输入图片为单通道，'6'表示输出通道数, ‘3’表示卷积核为3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.fc1 = nn.Linear(6 * 15 * 15, 10)

    # 正向传播
    def forward(self, x):
        # 卷积+池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.relu(x)
        # print(x.size())     # [1, 6, 14, 14]
        x = x.view(x.size()[0], -1)
        # print(x.size())     # [6, 16, 5, 5]
        x = self.fc1(x)

        return x


# for name, parameters in net.named_parameters():
#     print(name, ":", parameters.size())


if __name__ == '__main__':
    net = Net()
    print(net)

    x_in = torch.randn(1, 1, 32, 32)
    print(x_in.size())
    out = net(x_in)

    # net.zero_grad()
    # out.backward(torch.ones(1, 10))

    y = torch.arange(0, 10).view(1, 10).float()
    criterion = nn.MSELoss()
    loss = criterion(out, y)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())
