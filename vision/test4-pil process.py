from PIL import Image
import numpy as np
import torch
import cv2

path = 'E://BaiduNetdiskDownload//Face-Occlusion-Detect//data//cofw//face_train//20.jpg'
input_size = 64


def preprocess(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    pil_img = pil_img.resize((newW, newH))

    img_nd = np.array(pil_img)

    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)

    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans


if __name__ == '__main__':
    img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img2 = Image.open(path)
    print(img1.shape, img2.size)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    img1 = cv2.resize(img1, (input_size, input_size)) / 255
    img2 = preprocess(img2, 0.5)
    print(img1.shape, img2.shape)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    img1 = torch.from_numpy(img1).type(torch.FloatTensor)
    img2 = torch.from_numpy(img2).type(torch.FloatTensor)
    print(img1.shape, img2.shape)

