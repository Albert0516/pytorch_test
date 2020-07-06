import torch
import matplotlib.pylab as plt
from IPython import display


def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def xyplot(x_vals, y_vals, name):
    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')


if __name__ == '__main__':
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = x.relu()
    xyplot(x, y, 'relu')
