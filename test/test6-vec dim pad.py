import torch
import torch.nn.functional as F


# 矩阵维度扩展
values = torch.randn([2, 3, 3, 2])
print(values, values.shape)

padding_values = F.pad(values, [1, 2, 3, 4])
print(padding_values, padding_values.shape)
