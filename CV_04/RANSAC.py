import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
RANSAC 算法步骤：
0. while (dataset < k)  k 为迭代次数
1. 用4对点找到单应矩阵 H, 记为模型 M
2. 计算投影误差， 若小于阈值，则加入内点集 P
3. 若 |Pnew| > |P|,  P = Pnew, 更新 k
4. if (dataset > k) 停， or， continue

k = log(1 - p) / log(1 - w)
p 为自信力阈值(0.995)  w 为内点比例   m 为计算模型最少样本数量

什么是内点比例？

'''

def gen_data():
    data = np.random.randn(100, 2)

    data = 10 * data + 5
    return data

def RANSAC(data, k=10000, p=0.995):
    P = []
    nums, dim = data.shape
    init_ids = np.random.randint(0, nums, 8)
    src_pts = data[init_ids[:nums/2], :]
    dst_pts = data[init_ids[nums/2:], :]

    while data < k:
        M = cv2.findHomography(src_pts, dst_pts, None)
