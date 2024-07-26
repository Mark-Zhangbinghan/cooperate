import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm, trange   # trange(i)是tqdm(range(i))的一种简单写法

"""
X                                   数据集
nc                                  数据集的大小
alpha                               权衡参数
epsilon                             学习率

distance(x, y)                      计算两点之间距离

make_identity(nc)                   构建单位阵 I
make_adjacency(x, y)                构造邻接矩阵 A
make_degree(X)                      构造度矩阵 D
make_weight(A, M, alpha)            构建权重矩阵 W
make_r_weight(W)                    构建权重倒数矩阵 Wr
make_r_degree(Wr)                   构建权重合矩阵 Dr

make_instance(A)                    构建instance集合 instances
make_motif(A)                       构建motif矩阵 M
Algorithm_1(x, y, alpha)            收敛方法 MWMS-S
Algorithm_2(x, y, alpha)            收敛方法 MWMS-J
"""

# 1235v27v3qu6 rytw

def distance(x):  # 计算两点之间距离
    distances = np.sqrt((x[:, np.newaxis] - x) ** 2)
    return distances


def make_identity(nc):  # 构建单位阵I
    I = np.zeros((nc, nc))
    for i in range(nc):
        I[i, i] = 1
    return I


def make_adjacency(x):  # 构造邻接矩阵A
    distances = distance(x)
    nc = len(distances)
    A = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(nc):
            if distances[i, j] <= 1 and i != j:
                A[i, j] = 1
    return A


"""
A = np.where(distances <= 1, 1, 0)
for i in range(len(A):
    A[i, i] = 0
"""


def make_degree(A):  # 构造度矩阵D
    nc = A.shape[0]
    D = np.zeros(A.shape)
    for i in range(nc):
        D[i, i] = np.sum(A[i])
    return D


def make_weight(A, M, alpha):  # 构建权重矩阵W
    W = (1 - alpha) * A + alpha * M
    return W


def make_r_weight(W):  # 构建权重倒数矩阵Wr
    Wr = np.where(W != 0, 1 / W, 0)
    return Wr


def make_r_degree(Wr):  # 构建权重合矩阵Dr
    nc = len(Wr)
    Dr = np.zeros(Wr.shape)
    for i in range(nc):
        Dr[i, i] = np.sum(Wr[i])
    return Dr


def make_instance(A):  # 构建instance集合
    instances = []
    nc = len(A)
    for i in range(nc):
        for j in range(nc):
            if A[i, j] == 1:
                for k in range(nc):
                    if A[j, k] == 1 and A[i, k] == 1:
                        instances.append(sorted([i, j, k]))  # 对数据进行排序以便清除相同项
    instances = list(set(map(tuple, instances)))  # 清除相同项
    return instances


def make_motif(A):  # 构建motif矩阵M
    nc = len(A)
    instances = make_instance(A)
    M = np.zeros(A.shape)
    for i in range(nc):
        for j in np.arange(i+1, nc):
            if A[i, j] == 1:
                num = 0
                for k in range(len(instances)):
                    if (i in instances[k]) and (j in instances[k]):
                        num += 1
                M[i, j] = M[j, i] = num
    return M


def Algorithm_1(x, alpha):  # 收敛方法 MWMS-S
    try:
        A = make_adjacency(x)
        nc = len(A)
        I = make_identity(nc)
        D = make_degree(A)
        M = make_motif(A)
        W = make_weight(A, M, alpha)
        Wr = make_r_weight(W)
        Dr = make_r_degree(Wr)
        epsilon = np.where(D != 0, 1 / D, 0)
        x = (I - np.dot(epsilon, D - np.dot(np.dot(D, np.linalg.inv(Dr)), Wr))).dot(x)
        return x
    except:
        return x


def Algorithm_2(x, alpha):  # 收敛方法 MWMS-J
    A = make_adjacency(x)
    nc = len(A)
    I = make_identity(nc)
    M = make_motif(A)
    W = make_weight(A, M, alpha)
    Wr = make_r_weight(W)
    Dr = make_r_degree(Wr)
    x = np.dot(np.linalg.inv(I + Dr), (I + Wr)).dot(x)
    return x


def change_value(value_tabel, weight_matrix):
    tabel = list(set(tuple(value_tabel)))  # 清除相同项
    nc = len(tabel)
    matrix = np.zeros((len(weight_matrix), nc))
    index_map = {tup: i for i, tup in enumerate(tabel)}
    for i, weight in range(len(weight_matrix)), weight_matrix:
        for values in value_tabel:
            matrix[i][index_map[values]] = weight
    for col in range(matrix.shape[1]):  # 遍历列
        for _ in range(3):
            matrix[:, col] = Algorithm_2(matrix[:, col], 0.5)
    value_dict = {key: matrix[index_map[key]] for key in tabel}
    sorted_items = sorted(value_dict.items(), key=lambda item: item[1], reverse=True)
    # 选择排序后的列表中的前三个元素
    top_three_keys = [item[0] for item in sorted_items[:3]]
    return top_three_keys
