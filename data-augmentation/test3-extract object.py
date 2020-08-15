import matplotlib.pyplot as plt
import xml.dom.minidom as xmldom
import cv2

path_img = 'D:/CV group/competition-data/X ray/train/domain{}/{}.jpg'
path_xml = 'D:/CV group/competition-data/X ray/train/domain{}/XML/{}.xml'
save_path = 'D:/CV group/competition-data/X ray/objects/{}_{}.jpg'

labels = ['knife', 'scissors', 'lighter', 'zippooil', 'pressure', 'slingshot',
          'handcuffs', 'nailpolish', 'powerbank', 'firecrackers']

object_id = 9

for folder_id in range(1, 7):
    fid = folder_id * 100000
    for i in range(fid+1, fid+601):
        stri = str(i)
        print('------'+str(i)+'------')
        image = cv2.imread(path_img.format(folder_id, stri))
        if image is None:
            continue

        domTree = xmldom.parse(path_xml.format(folder_id, stri))
        rootNode = domTree.documentElement
        objects = rootNode.getElementsByTagName('object')

        sub_i = 1
        for object in objects:
            name = object.getElementsByTagName('name')[0].childNodes[0].data
            if name != labels[object_id]:
                continue
            print(name)
            box = object.getElementsByTagName('bndbox')[0]

            xmin = int(box.getElementsByTagName('xmin')[0].childNodes[0].data.split('.')[0])
            xmax = int(box.getElementsByTagName('xmax')[0].childNodes[0].data.split('.')[0])
            ymin = int(box.getElementsByTagName('ymin')[0].childNodes[0].data.split('.')[0])
            ymax = int(box.getElementsByTagName('ymax')[0].childNodes[0].data.split('.')[0])

            print(ymin, ymax, xmin, xmax)
            image = image[ymin:ymax, xmin:xmax, :]           # 裁剪
            if image.shape[0] != 0 and image.shape[1] != 0:
                cv2.imwrite(save_path.format(stri, str(sub_i)), image)
                sub_i = sub_i+1






