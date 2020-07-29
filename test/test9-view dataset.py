import numpy as np
import matplotlib.pyplot as plt
import os

# os.listdir('D://AI调研//比赛&数据集//youtube_faces_with_keypoints_full_1//')
path = 'D://CV group//competition-data//youtube_faces_with_keypoints_full_1//Adrian_Nastase_4.npz'
# 数据格式-['colorImages', 'boundingBox', 'landmarks2D', 'landmarks3D']
data = np.load(path)

# (179, 144, 3, num_frames) -> (num_frames, 179, 144, 3)
images = data['colorImages'].transpose((3, 0, 1, 2))
box = data['boundingBox'].transpose((2, 0, 1))          # (num_frames, 4, 2)
marks2d = data['landmarks2D'].transpose(2, 0, 1)        # (num_frames, 68, 2)
marks3d = data['landmarks3D'].transpose(2, 0, 1)        # (num_frames, 68, 3)

plt.ion()
for i in range(len(images)):
    image = images[i]
    x = marks2d[i, :, 0]
    y = marks2d[i, :, 1]

    plt.imshow(image)
    plt.scatter(x, y, marker='*', color='r')
    plt.draw()
    # plt.show()
    plt.pause(0.5)
    plt.clf()

