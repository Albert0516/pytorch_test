import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
from torchvision import models, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim

# picture visualization test
# cat_img = Image.open('../Felis_silvestris_catus_lying_on_rice_straw.jpg')
# print(cat_img.size)
#
# transform_224 = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
# ])
# cat_img_224 = transform_224(cat_img)
#
# writer = SummaryWriter(log_dir='E:/tensorboard_logs', comment='cat image')
# writer.add_image('cat', cat_img_224)
# writer.close()

BATCH_SIZE = 512
EPOCHS = 20
train_loader = Data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv2d(1, 10, 5)    # 10, 24x24
        self.conv2 = nn.Conv2d(10, 20, 3)   # 128, 10x10
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)     # 24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)   # 12
        out = self.conv2(out)   # 10
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet()
optimizer = optim.Adam(model.parameters())


def train(model, train_loader, optimizer, epoch):
    n_iter = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1) % 30 == 0:
            n_iter = n_iter+1
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # 相对于以前的训练方法 主要增加了以下内容
            out = torch.cat((output.data, torch.ones(len(output), 1)), 1)       # 因为是投影到3D的空间，所以我们只需要3个维度
            with SummaryWriter(log_dir='E:/tensorboard_logs', comment='mnist') as writer:
                # 使用add_embedding方法进行可视化展示
                writer.add_embedding(
                    out,
                    metadata=target.data,
                    label_img=data.data,
                    global_step=n_iter)


if __name__ == '__main__':
    train(model, train_loader, optimizer, 0)
