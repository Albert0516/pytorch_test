import os

PATH = 'E://PyCharm_project//Pytorch-UNet//data'


# 数据整理（图片&标签匹配）,贪心算法
def list_dir(folder1, folder2):
    list1 = os.listdir(PATH + folder1)
    list2 = os.listdir(PATH + folder2)
    print(list1)
    print(list2)

    i = 0
    while i < len(list1):
        tmp1 = list1[i].split('.')[0]
        tmp2 = list2[i].split('.')[0]

        if tmp1 == tmp2:
            i = i + 1
        else:
            print(list1[i] + "---" + list2[i])
            os.remove(PATH + folder2 + '//' + list2[i])
            list2.pop(i)

    while i < len(list2):
        print("list1 over ---" + list2[i])
        os.remove(PATH + folder2 + '//' + list2[i])
        list2.pop(i)


if __name__ == '__main__':
    list_dir('//masks', '//imgs')
