import torch

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("gpu status:", device)
    if torch.cuda.is_available():
        print("pytorch version:", torch.__version__, "| device name:", torch.cuda.get_device_name())
        print("gpu nums:", torch.cuda.device_count())

        x = torch.Tensor([1.0])
        xx = x.cuda()               # method1:transfer to cuda
        xx_gpu = x.to(device)       # method2:transfer to cuda
        print(xx, xx_gpu)

        y = torch.randn(2, 3)
        yy = y.cuda()
        print(yy)

    from torch.backends import cudnn
    print("Support cudnn ?:", cudnn.is_acceptable(xx))
