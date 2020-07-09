import h5py
import numpy as np
import time
import matplotlib.pyplot as plt

file_path = 'E://BaiduNetdiskDownload//Face-Occlusion-Detect//data//cofw//COFW_train.mat'

data = h5py.File(file_path, 'r')
print(list(data.keys()))

# images = np.transpose(data['IsTr'])
# print(images.shape)
# plt.imshow(images[0])
# plt.show()

phisTr = np.transpose(data['phisTr'])
index = 10
x = np.array(phisTr[index][0:29])
y = np.array(phisTr[index][29:58])
label = np.array(phisTr[index][58:])

plt.scatter(x, y, marker='*', color='r')
plt.show()
# for label in phisTr:
#     print(label)
#     time.sleep(1)



