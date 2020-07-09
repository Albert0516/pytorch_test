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


x = np.array(x)
y = np.array(y)
print(x, len(x.shape))
print(y, len(y.shape))

x = imgplt.imread('E://test_data//image_0002.png')
plt.imshow(x)

plt.scatter(x, y, marker='*', color='r')
plt.show()

# plt.waitforbuttonpress(0)
