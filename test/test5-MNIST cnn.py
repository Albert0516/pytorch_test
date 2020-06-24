import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as Data

BATCH_SIZE = 512
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 训练集
train_loader = Data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)

# 测试集
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)

        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        # 第一层卷积+池化(kernal=5) [1, 1, 28, 28]->[1, 10, 24, 24]->[1, 10, 12, 12]
        out_conv1 = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        # 第二层卷积(kernal=3) [1, 10, 12, 12] -> [1, 20, 10, 10]
        out_conv2 = F.relu(self.conv2(out_conv1))
        out1 = out_conv2.view(in_size, -1)
        out2 = F.relu(self.fc1(out1))
        out = self.fc2(out2)
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())


def train(model, device, train_loader_data, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader_data):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_data.dataset),
                100. * batch_idx / len(train_loader_data), loss.item()
            ))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()     # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]                               # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


if __name__ == '__main__':
    for epoch in range(1, EPOCHS+1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)
