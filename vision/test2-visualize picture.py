import matplotlib.image as imgplt
import matplotlib.pyplot as plt
import numpy as np

path = 'E://test_data//image_00'
pts = '.pts'
png = '.png'

for index in range(1, 31):
    x, y = [], []
    other = ''
    if index < 10:
        other = '0'
    with open(path + other + str(index) + pts, 'r') as f:
        axis = f.readlines()
        for i in range(3, 71):
            tmp = axis[i].split(' ')
            x.append(float(tmp[0]))
            y.append(float(tmp[1][:10]))

    x = np.array(x)
    y = np.array(y)
    # print(x, len(x.shape))
    # print(y, len(y.shape))

    image = imgplt.imread(path + other + str(index) + png)
    plt.scatter(x, y, marker='.', color='r')

    plt.imshow(image)
    plt.draw()
    plt.pause(0.5)
    plt.clf()

# plt.waitforbuttonpress(0)
