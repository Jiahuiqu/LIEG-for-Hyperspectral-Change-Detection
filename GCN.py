#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:10:12 2021

@author: xidian
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:1")
class GCNLayer(nn.Module):
    """
    input_dim: 输入维度
    output_dim：输出维度
    A：　　　　　邻接矩阵
    """

    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayer, self).__init__()
        self.A = A + torch.eye(A.shape[0], A.shape[0], requires_grad=False).to(device)
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # 第一层GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        nodes_count = self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)

        self.mask = torch.ceil(self.A * 0.00001)

    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        H = self.BN(H)
        #H_xx1 = self.GCN_liner_theta_1(H)  # size = 256
        # e = torch.sigmoid(torch.matmul(H_xx1, H_xx1.t()))
        # zero_vec = -9e15 * torch.ones_like(e)
        # A = torch.where(self.mask > 0, e, zero_vec) + self.I
        # if model != 'normal': A = torch.clamp(A, 0.1)  # This is a trick for the Indian Pines.
        # A = F.softmax(A, dim=1)
        output = self.Activition(torch.mm(self.A, self.GCN_liner_out_1(H)))
        return output

class CEGCN(nn.Module):
    def __init__(self, class_count: int, Q1: torch.Tensor, A1: torch.Tensor, model='normal'):
        """
        :param height: 图像的高度
        :param width:  图像的宽度
        :param channel:  图像的通道数
        :param class_count:
        :param Q1: 原图像1和聚类的图像的映射关系
        :param A1: 图结构的邻接矩阵
        :param Q2: 原图像2和聚类的图像的映射关系
        :param A2：图结构的邻接矩阵
        """
        super(CEGCN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        # self.channel = channel
        # self.height = height
        # self.width = width
        self.Q1 = Q1
        self.A1 = A1
        self.model = model
        self.norm_col_Q1 = torch.sum(Q1, 0, keepdim=True) # 列归一化Q

        layers_count_d = 0  # 三轮的图卷积
        layers_count_g = 5


        # # Spectra Transformation Sub-Network
        # self.CNN_denoise = nn.Sequential()
        # for i in range(layers_count_d):
        #     if i == 0:
        #         self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
        #         self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
        #                                     nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
        #         self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        #     else:
        #         self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
        #         self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
        #         self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        # Superpixel-level Graph Sub-Network
        self.GCN_Branch1 = nn.Sequential()
        for i in range(layers_count_g):
            if i != layers_count_g - 1:
                self.GCN_Branch1.add_module('GCN_Branch1' + str(i), GCNLayer(256, 256, self.A1))
            else:
                self.GCN_Branch1.add_module('GCN_Branch1' + str(i), GCNLayer(256, 256, self.A1))


        """
        最后的GCN输出的是类别
        """
        # Softmax layer
        self.Softmax_linear = nn.Sequential(nn.Linear(256, self.class_count))
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x1: torch.Tensor):
        """
        :param x2:
        :param x1:
        :return: probability_map
        """
        # (h1, w1, c1) = x1.shape
        # norm_col_Q1 = torch.sum(self.Q1, 0, keepdim = True)
        # # 先去除噪声GCN_Branch1
        # noise = self.CNN_denoise(torch.unsqueeze(x1.permute([2, 0, 1]), 0))
        # noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        # clean_x = noise  # 直连
        
        # clean_x_flatten = clean_x.reshape([h1 * w1, -1])
        
        
        
        superpixels_flatten_1 = x1
        
        
        
        # print((self.Q1/1.0).t().shape,superpixels_flatten_1.shape)
        # superpixels_flatten_1 = torch.mm(self.Q1.t(), superpixels_flatten_1)  # 低频部分

        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q
        norm_col_Q1 = torch.sum(self.Q1, 0, keepdim=True)
        H1 = superpixels_flatten_1 / norm_col_Q1.t()
        if self.model == 'normal':
            for i in range(len(self.GCN_Branch1)):
                H1 = self.GCN_Branch1[i](H1)
        # print(self.Q1.float().shape,x1.shape,H1.shape)
        # GCN_result = torch.matmul(self.Q1.float(), H1.float())  # 这里self.norm_row_Q == self.Q
        Y = self.softmax(self.Softmax_linear(H1))
    
        
        
        return Y
    
    
    
    
if __name__ == "__main__":
    Q = torch.randint(0,2 ,size=(58880, 4230)).cuda()
    A = torch.randint(0,2,size=(4230,4230)).cuda()
    GCN = CEGCN(3, Q1=Q, A1 = A).cuda()
    out = torch.rand(4230,256).cuda()
    out = GCN(out)
    print(out.shape)
    
    
    
    
    
