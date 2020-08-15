import torch

x = torch.randn(100, 1, 96, 96)
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
conv1_1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)


max2d = torch.nn.MaxPool2d(kernel_size=2)
max3d = torch.nn.MaxPool2d(kernel_size=3)
bth32 = torch.nn.BatchNorm2d(32)
bth64 = torch.nn.BatchNorm2d(64)
bth128 = torch.nn.BatchNorm2d(128)
bth256 = torch.nn.BatchNorm2d(256)

bth1 = torch.nn.BatchNorm1d(512)
bth2 = torch.nn.BatchNorm1d(128)
relu = torch.nn.ReLU()
drop = torch.nn.Dropout(0.5)

fcnn1 = torch.nn.Linear(8*8*128, 512)
fcnn2 = torch.nn.Linear(512, 128)
fcnn3 = torch.nn.Linear(128, 6)


h1 = conv1(x)
h1 = bth32(h1)
h1 = relu(h1)
h1 = conv1_1(h1)
h1 = bth32(h1)
h1 = relu(h1)
h1 = max3d(h1)
print(h1.shape)

h2 = conv2(h1)
h2 = bth64(h2)
h2 = relu(h2)
h2 = max2d(h2)
print(h2.shape)

h3 = conv3(h2)
h3 = bth128(h3)
h3 = relu(h3)
h3 = max2d(h3)
print(h3.shape)


out1 = fcnn1(h3.view(h3.size(0), -1))
out1 = bth1(out1)
out1 = relu(out1)
out1 = drop(out1)
print(out1.shape)
out2 = fcnn2(out1)
out2 = bth2(out2)
out2 = relu(out2)
out2 = drop(out2)
print(out2.shape)
out = fcnn3(out2)
print(out.shape)


