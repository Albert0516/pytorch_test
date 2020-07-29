# coding=utf-8
import torch

# 测试pytorch-GPU是否能用

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pytorch version:", torch.__version__, "| device status:", device)

    x = torch.Tensor([1.0])
    xx = x.cuda()               # method1:transfer to cuda
    xx_gpu = x.to(device)       # method2:transfer to cuda
    print(xx, xx_gpu)

    y = torch.randn(2, 3)
    yy = y.cuda()
    print(yy)

    zz = xx + yy
    print(zz)

    from torch.backends import cudnn
    print("Support cudnn ?:", cudnn.is_acceptable(xx))
