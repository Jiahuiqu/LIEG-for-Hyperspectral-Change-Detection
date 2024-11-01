#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:10:42 2021

@author: xidian
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.io as sio


def SegmentsLabelProcess(labels):
    """
    对labels做后处理，防止出现label不连续现象
    """
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLIC(object):
    def __init__(self,  img1,  n_segments=1000, compactness=5, max_iter=20, sigma=0, min_size_factor=0.8,
                 max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma
        self.img1 = img1
        # 数据standardization标准化,即提前全局BN
        height, width, bands = self.img1.shape  # 原始高光谱数据的三个维度
        data = np.reshape(self.img1, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.fuse = np.reshape(data, [height, width, bands])

    def get_Q_and_S_and_Segments(self):
        # 执行 SLIC 并得到Q(nxm),S(m*b)
        (h1, w1, d1) = self.img1.shape
        # 计算超像素S以及相关系数矩阵Q segments 610 * 340
        segments = slic(self.img1, n_segments=self.n_segments, compactness=self.compactness,
                        convert2lab=False, sigma=self.sigma, enforce_connectivity=True,
                        min_size_factor=self.min_size_factor, max_size_factor=self.max_size_factor, slic_zero=False)

        """
        判断超像素label是否连续,否则予以校正
        superpixel_count 表示最后分出来的超像素点的个数
        """
        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))): segments = SegmentsLabelProcess(
            segments)
        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)

        """
        显示超像素的图片
        """
        out = mark_boundaries(self.img1[:, :, [5, 15, 20]], segments)
        plt.figure()
        plt.imshow(out)  # 读取out，但不显示
        plt.show()  # 显示

        """
        image1的S、Q、X 
        """
        segments = np.reshape(segments, [-1])
        S1 = np.zeros([superpixel_count, d1], dtype=np.float32)
        Q1 = np.zeros([w1 * h1, superpixel_count], dtype=np.float32)
        x1 = np.reshape(self.img1, [-1, d1])
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x1[idx]
            superpixel = np.sum(pixels, 0) / count
            S1[i] = superpixel
            Q1[idx, i] = 1

        self.S1 = S1
        self.Q1 = Q1


        return Q1, S1, superpixel_count

    def get_A(self, sigma: float):
        """
         根据 segments 判定邻接矩阵
        :return:
        """
        A1 = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A1[idx1, idx2] != 0:
                        continue

                    pix1 = self.S1[idx1]
                    pix2 = self.S1[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A1[idx1, idx2] = A1[idx2, idx1] = diss


        return A1


class LDA_SLIC(object):
    def __init__(self, data1, labels, n_component):
        """
        :param data1: 数据1
        :param data2: 数据2
        """
        self.data1 = data1
        self.init_labels = labels
        self.n_component = n_component
        self.height, self.width, self.bands = data1.shape
        self.x_flatt = np.reshape(data1, [self.width * self.height, self.bands])
        self.y_flatt = np.reshape(labels, [self.height * self.width])
        self.labels = labels

    def LDA_Process(self, curr_labels):  # LDA降维，其为有监督降维，这里为多类别LDA降维
        """
        :param curr_labels: height * width
        :return:
        """
        curr_labels = np.reshape(curr_labels, [-1])
        idx = np.where(curr_labels != 0)[0]
        x = self.x_flatt[idx]
        y = curr_labels[idx]
        lda = LinearDiscriminantAnalysis()  # n_components=self.n_component
        lda.fit(x, y)  # LDA模型只用了训练集
        X_new = lda.transform(self.x_flatt)  # 全图做LDA，Pavia降到了8维
        return np.reshape(X_new, [self.height, self.width, -1])

    def SLIC_Process(self, img1,scale=25):
        """
        superpixel_count 2308
        Q 207400 * 2308  输入图和超像素之间的关系
        S 2308 * 8       初始化A的时候使用
        A 2308 * 2308    邻接矩阵
        seg 610 * 340    后面没有使用
        """
        n_segments_init = self.height * self.width / scale
        print("n_segments_init", n_segments_init)
        myslic = SLIC(img1, n_segments=n_segments_init, compactness=20, sigma=1, min_size_factor=0.2,
                      max_size_factor=3)
        Q1, S1, superpixel_count = myslic.get_Q_and_S_and_Segments()
        A1 = myslic.get_A(sigma=10)
        return Q1, S1, A1, superpixel_count

    def simple_superpixel(self, scale):
        curr_labels = self.init_labels
        X = self.LDA_Process(curr_labels)
        Q1, S1, A1,  superpixel_count = self.SLIC_Process(self.data1, scale=scale)
        return Q1, S1, A1, superpixel_count


