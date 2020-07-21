import numpy as np


# 创建一个10*10的ndarray对象，且矩阵边界全为1，里面全为0
def create_array():
    nd = np.zeros(shape=(10, 10), dtype=np.int8)
    nd[[0, 9]] = 1
    nd[:, [0, 9]] = 1
    print(nd)


# 创建一个长度为10的随机数组并排序
def sort():
    a8 = np.random.random(10)
    a8_sort = a8.argsort()
    print(a8, a8_sort, a8[a8_sort])


# 根据某一列对矩阵排序
def sort_by_column(m, n, i):
    temp = np.random.randint(0, 100, size=(m, n))
    print(temp)
    print(temp[np.argsort(temp[:, i])])


# 给定数组，将该数组按间隔index插入新数组
def insert_array(i, n):
    a = np.arange(1, n)
    b = np.zeros(shape=i*(len(a)-1) + len(a), dtype=int)
    b[::i+1] = a
    print(b)


# 交换二维数组的行/列
def exchage():
    a = np.random.randint(0, 100, size=(3, 3))
    print(a)
    print(a[[2, 0, 1]])
    print(a[:, [2, 0, 1]])


# 矩阵减去其按行均值
def minus_mean():
    a = np.random.randint(0, 10, (3, 3))
    print(a)
    b = a.mean(axis=1).reshape(3, 1)
    print(a - b)


# 间隔插值
def insert_one():
    a = np.zeros(shape=(8, 8), dtype=int)
    a[::2, 1::2] = 1
    a[1::2, ::2] = 1
    print(a)


# 正则化矩阵
def normalization():
    a = np.random.randint(0, 100, size=(5, 5))
    print(a)
    a_min = a.min()
    a_max = a.max()
    a = (a-a_min)/(a_max-a_min)
    print(a)


# 冒泡排序
def bubble_sort(a):
    print(a)
    b = a[0]
    for i in range(len(b)):
        for j in range(1, len(b)-i):
            if b[j] < b[j-1]:
                b[j], b[j-1] = b[j-1], b[j]
    print(a)


if __name__ == '__main__':
    # sort_by_column(6, 6, 4)
    # insert_array(4, 7)
    # exchage()
    # minus_mean()
    # normalization()
    bubble_sort(np.random.randint(0, 100, size=(1, 20)))



