import numpy as np
import torch
from torch import nn

input = torch.randn(3, 3)
print(input)

m = nn.Sigmoid()
out = m(input)
print(out)

target = torch.FloatTensor([[0, 1, 1],
                            [0, 0, 1],
                            [1, 0, 1]])

loss = nn.BCELoss()
print(loss(out, target))
