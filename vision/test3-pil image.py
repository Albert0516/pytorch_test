from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
import cv2


def image_open():
    image_path = "E://BaiduNetdiskDownload//Face-Occlusion-Detect//data//cofw//face_train//20.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64)) / 255
    # image = Image.open(image_path)
    # 转成numpy数组形式
    image_array = np.array(image)
    print("image_array:", image_array.shape)

    image_new = torch.Tensor(image_array)
    print("image_new:", image_new.shape)
    # image.show()

    input_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = input_transform(image).unsqueeze(0)
    print("image_tensor:", image.shape)


if __name__ == '__main__':
    # label = [[1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]]
    # print(label)
    # label = np.array(label)
    # print(label.shape)
    image_open()
