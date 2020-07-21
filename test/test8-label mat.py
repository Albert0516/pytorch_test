import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = 'E://BaiduNetdiskDownload//Face-Occlusion-Detect//data//cofw//COFW_train.mat'

data = h5py.File(file_path, 'r')            # ['#refs#', 'IsTr', 'bboxesTr', 'phisTr']
images = np.transpose(data['IsTr'])         # (1345, 1)
labels = np.transpose(data['phisTr'])       # (1345, 87)

plt.ion()
for index in range(len(images)):
    # 理解！
    image = np.transpose(data[images[index][0]])
    label = labels[index]
    x, x_occluded = [], []
    y, y_occluded = [], []
    for i in range(58, len(label)):
        if label[i] == 1:
            x_occluded.append(label[i-58])
            y_occluded.append(label[i-29])
        else:
            x.append(label[i-58])
            y.append(label[i-29])

    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y, marker='*', color='b')

    if len(x_occluded) > 0:
        x_occluded = np.array(x_occluded)
        y_occluded = np.array(y_occluded)
        plt.scatter(x_occluded, y_occluded, marker='*', color='r')

    plt.imshow(image)
    plt.draw()
    plt.pause(0.5)
    plt.clf()




