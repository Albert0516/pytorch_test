import torch
from time import time
# 运行时间测试

a = torch.ones(1000)
b = torch.ones(1000)

start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time()-start)

start = time()
d = a + b           # 矢量计算
print(time()-start)
