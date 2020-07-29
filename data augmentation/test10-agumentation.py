import cv2
import argparse
import xml.dom.minidom as xmldom
import numpy as np
import random
import os

object_path = "D:/CV group/competition-data/X ray/objects/"

path_img = 'D:/CV group/competition-data/X ray/train/domain{}/{}.jpg'
path_xml = 'D:/CV group/competition-data/X ray/train/domain{}/XML/{}.xml'

save_xml = 'D:/CV group/competition-data/X ray/augmentation/XML/{}.xml'
save_img = 'D:/CV group/competition-data/X ray/augmentation/{}.jpg'

# labels_all = ['knife', 'scissors', 'lighter', 'zippooil', 'pressure', 'slingshot',
#           'handcuffs', 'nailpolish', 'powerbank', 'firecrackers']

labels = [[['zippooil', 'nailpolish']],
          [['knife', 'slingshot', 'handcuffs']]]
object_class = 2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="D:/CV group/competition-data/X ray/augmentation/",
                        help='path for saving the processed images and xml files')
    parser.add_argument('--image_dir', type=str, default="D:/CV group/competition-data/X ray/train/domain{}/",
                        help='path for loading the origin image')
    parser.add_argument('--object_dir', type=str, default="D:/CV group/competition-data/X ray/objects/",
                        help='path for loading the images of object')
    parser.add_argument('--class', type=int, default=1,
                        help='class for 1 (nailpolish and zippooil) or 2(knife, slingshot and handcuffs)')
    parser.add_argument('--epoch', type=int, default=10,
                        help='amount of proposed data = epoch*6 ')

    return parser.parse_args()


# 向目标xml文件添加label坐标信息
def add_label_xml(axis, l_name):
    domTree = xmldom.parse(path_xml.format(domain_id, img_name))
    rootNode = domTree.documentElement              # 根节点

    for i in range(len(l_name)):
        object_node = domTree.createElement("object")     # 根节点下新建一级子节点object
        # 二级子节点-name
        name_node = domTree.createElement("name")
        name_value = domTree.createTextNode(l_name[i])
        name_node.appendChild(name_value)
        object_node.appendChild(name_node)
        # 二级子节点-pose
        pose_node = domTree.createElement("pose")
        pose_value = domTree.createTextNode("Unspecified")
        pose_node.appendChild(pose_value)
        object_node.appendChild(pose_node)
        # 二级子节点-truncated
        truncated_node = domTree.createElement("truncated")
        truncated_value = domTree.createTextNode("0")
        truncated_node.appendChild(truncated_value)
        object_node.appendChild(truncated_node)
        # 二级子节点-difficult
        difficult_node = domTree.createElement("difficult")
        difficult_value = domTree.createTextNode("0")
        difficult_node.appendChild(difficult_value)
        object_node.appendChild(difficult_node)
        # 二级子节点-check
        check_node = domTree.createElement("check")
        check_value = domTree.createTextNode("0")
        check_node.appendChild(check_value)
        object_node.appendChild(check_node)
        # 二级子节点-bndbox
        bndbox_node = domTree.createElement("bndbox")
        # bndbox下三级子节点(bndbox有四个子节点:xmin, ymin, xmax, ymax)
        xmin_node = domTree.createElement("xmin")
        xmin_value = domTree.createTextNode(str(axis[i][0]))
        xmin_node.appendChild(xmin_value)
        bndbox_node.appendChild(xmin_node)

        ymin_node = domTree.createElement("ymin")
        ymin_value = domTree.createTextNode(str(axis[i][1]))
        ymin_node.appendChild(ymin_value)
        bndbox_node.appendChild(ymin_node)

        xmax_node = domTree.createElement("xmax")
        xmax_value = domTree.createTextNode(str(axis[i][2]))
        xmax_node.appendChild(xmax_value)
        bndbox_node.appendChild(xmax_node)

        ymax_node = domTree.createElement("ymax")
        ymax_value = domTree.createTextNode(str(axis[i][3]))
        ymax_node.appendChild(ymax_value)
        bndbox_node.appendChild(ymax_node)

        object_node.appendChild(bndbox_node)
        rootNode.appendChild(object_node)

    with open(save_xml.format(str(img_name)), 'w') as f:
        domTree.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')


# 设置所添加的label坐标
def generate_axis(image, imgs):
    # print(image.shape, img_label.shape)
    result = []
    for label_img in imgs:
        x = label_img.shape[1]
        y = label_img.shape[0]
        x_min = random.randint(0, image.shape[1] - x)
        y_min = random.randint(0, image.shape[0] - y)
        print(x_min, y_min, x_min+x, y_min+y)
        result.append([x_min, y_min, x_min+x, y_min+y])
    return result


# 将目标添加到图像中
def add_label_img(axis, images, label, label_gray):
    for k in range(len(axis)):
        # 边缘检测
        _, thresh = cv2.threshold(label_gray[k], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # edges = cv2.Canny(img, 0, 255)  # canny边缘检测
        # 生成掩膜
        mask = np.array(thresh)
        target = np.where(mask == 0)
        for i, j in zip(target[0], target[1]):
            images[axis[k][1] + i][axis[k][0] + j] = label[k][i][j]

        # image = cv2.GaussianBlur(image, (5, 5), 0)
        cv2.imwrite(save_img.format(str(img_name)), images)


if __name__ == '__main__':
    for i in range(101):
        print("-------turn:{}--------".format(i))
        for domain_id in range(1, 7):
            seed = random.randint(1, 601)
            img_id = domain_id * 100000
            img_name = img_id + seed
            print(path_img.format(domain_id, img_name))
            img = cv2.imread(path_img.format(domain_id, img_name))

            label_imgs = []
            label_grays = []
            if img is None or img.shape[0] <= 0 or img.shape[1] <= 0: continue
            for label_name in labels[object_class]:
                folder = object_path + label_name
                dirs = os.listdir(folder)
                label = dirs[random.randint(0, len(dirs)-1)]
                path = folder + '/' + label
                object_img = cv2.imread(path)
                if object_img is None or object_img.shape[0] <= 0 or object_img.shape[1] <= 0: continue
                if img.shape[0]-object_img.shape[0] <= 0 or img.shape[1]-object_img.shape[1] <= 0:
                    continue
                label_imgs.append(object_img)
                label_grays.append(cv2.imread(path, 0))
            if len(label_imgs) == len(labels[object_class]):
                axis = generate_axis(img, label_imgs)
                add_label_img(axis, img, label_imgs, label_grays)
                add_label_xml(axis, labels[object_class])
