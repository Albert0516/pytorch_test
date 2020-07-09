import matplotlib.image as imgplt
import matplotlib.pyplot as plt
import numpy as np

x, y = [], []

with open('E://test_data//image_0002.pts', 'r') as f:
    axis = f.readlines()
    for i in range(3, 71):
        tmp = axis[i].split(' ')
        x.append(float(tmp[0]))
        y.append(float(tmp[1][:10]))


x_data = np.array(x)
y_data = np.array(y)
print(x_data, len(x_data.shape))
print(y_data, len(y_data.shape))

x = imgplt.imread('E://test_data//image_0002.png')
plt.imshow(x)

plt.scatter(x_data, y_data, marker='*', color='r')
plt.show()

# plt.waitforbuttonpress(0)
