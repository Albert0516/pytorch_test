import numpy as np
import matplotlib.pyplot as plt

data = np.load('D://AI调研//比赛&数据集//youtube_faces_with_keypoints_full_1//Aaron_Eckhart_1.npz')
print(data.files)
image = data['colorImages']
print(image.shape)

# plt.imshow(np.transpose(image))
