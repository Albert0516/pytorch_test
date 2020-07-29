'''
cd Anaconda3/Scripts/
pip install opencv-python
pip install opencv-contrib-python
'''

import cv2
import numpy as np


# 用于显示的函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图片路径
pth = ''

# 读取图片
img = cv2.imread(pth)
# # 如果路径中含有中文
# img = cv2.imdecode(np.fromfile(pth, dtype=np.uint8), 1)
# 显示
cv_show('Original', img)

# 坐标
x0, y0, x1, y1 = 0, 1, 0, 1

# 裁剪
img = img[y0:y1, x0:x1, :]
# 显示
cv_show('Croped', img)

# 保存
newPath = ''                                        # 新路径
cv2.imwrite(newPath, img)
# # 如果路径中含有中文
# cv2.imencode('.jpg', img)[1].tofile(newPath)
