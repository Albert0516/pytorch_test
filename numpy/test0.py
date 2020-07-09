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


if __name__ == '__main__':
    sort_by_column(6, 6, 4)
