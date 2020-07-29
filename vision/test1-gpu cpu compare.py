import torch
from time import time

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pytorch version:", torch.__version__, "| device status:", device)

    start = time()
    x = torch.Tensor([1.0])
    y = torch.randn(2, 3)
    z = x + y
    print('result_cpu = {}|time = {}'.format(z, time()-start))

    start = time()
    xx = x.cuda()
    yy = y.cuda()
    zz = xx+yy
    print('result_gpu = {}|time = {}'.format(zz, time() - start))