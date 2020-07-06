import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")
# import d2lzh_pytorch as d2l

minst_train = torchvision.datasets.FashionMNIST()
