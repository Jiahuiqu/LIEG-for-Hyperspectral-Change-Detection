#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:10:27 2021

@author: xidian
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
from model import D_LIEG
import SLIC
from AdjMatrix import AdjMatrix
# from OA_KAPPA import *

root = '/bay/250/Norm/'
def main():
    device = torch.device('cuda:0')
    samples_type = ['ratio', 'same_num'][0]  # 比例还是固定的数
    """
    FLAG =1, indian
    FLAG =2, Bay
    FLAG =3, salinas
    curr_train_ratio 为训练集比例
    Scale 定义超像素的相关 
    """
    for (FLAG, curr_train_ratio, Scale) in [(2, 0.005, 150)]:
        torch.cuda.empty_cache()
        OA_ALL = []
        AA_ALL = []
        KPP_ALL = []
        AVG_ALL = []
        Train_Time_ALL = []
        Test_Time_ALL = []

        Seed_List = [0]  # 随机种子点
        class_count = 0

        if FLAG == 2:
            """
            data1: T1时刻的数据
            data2: T2时刻的数据
            gt:    标签
            """
            data_mat1 = sio.loadmat(root+'bay_t1.mat')
            data_mat2 = sio.loadmat(root+'bay_t2.mat')
            data1 = data_mat1['Q1']
            data2 = data_mat2['Q2']
            gt_mat = sio.loadmat(root+'bay_gt.mat')
            gt = gt_mat['gt']

            # 参数预设
            train_ratio = 0.005  # 训练集比例。注意，训练集为按照‘每类’随机选取
            val_ratio = 0.00625  # 测试集比例.注意，验证集选取为从测试集整体随机选取，非按照每类
            class_count = 2  # 变化检测样本类别数
            learning_rate = 0.00001  # 学习率
            max_epoch = 200  # 迭代次数 1000 - 3000
            dataset_name = "Bar"  # 数据集名称
            pass

        superpixel_scale = Scale
        train_samples_per_class = curr_train_ratio  # 当定义为每类样本个数时,则该参数更改为训练样本数
        val_samples = class_count  # 样本类别数
        train_ratio = curr_train_ratio  # 训练比例
        cmap = cm.get_cmap('jet', class_count + 1)
        # plt.set_cmap(cmap)
        m, n, d = data1.shape  # 高光谱数据的三个维度

        # 数据standardization标准化,即提前全局BN
        height, width, bands = data1.shape  # 原始高光谱数据的三个维度
        # print(height, width, bands)
        data1 = np.reshape(data1, [height * width, bands])  # 将数据转为HW * B
        minMax = preprocessing.StandardScaler()
        data1 = minMax.fit_transform(data1)  # 这两行用来归一化数据，归一化时需要进行数据转换
        data1 = np.reshape(data1, [height, width, bands])  # 将数据转回去 H * W * B
        data2 = np.reshape(data2, [height * width, bands])  # 将数据转为HW * B
        minMax = preprocessing.StandardScaler()
        data2 = minMax.fit_transform(data2)  # 这两行用来归一化数据，归一化时需要进行数据转换
        data2 = np.reshape(data2, [height, width, bands])  # 将数据转回去 H * W * B

        # 打印每类样本个数
        gt_reshape = np.reshape(gt, [-1])
        for i in range(class_count+1):
            idx = np.where(gt_reshape == i)[-1]
            samplesCount = len(idx)
            print(samplesCount)



        def GT_To_One_Hot(gt, class_count):
            """
            Convet Gt to one-hot labels
            :param gt:
            :param class_count:
            :return:
            """
            GT_One_Hot = []  # 转化为one-hot形式的标签
            for i in range(gt.shape[0]):
                for j in range(gt.shape[1]):
                    temp = np.zeros(class_count, dtype=np.float32)
                    if gt[i, j] != 0:
                        temp[int(gt[i, j]-1)] = 1
                    GT_One_Hot.append(temp)
            GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
            return GT_One_Hot



        def image_into_patch(image_T1, image_T2, train_data_index, class_count, superpixel_scale):
            """
            transform T1 and T2 into patch
            """
            height, width, band = image_T1.shape
            img = np.concatenate((image_T1, image_T2), axis=-1)  ##T1和T2级联
            ls = SLIC.LDA_SLIC(img, np.reshape(train_samples_gt, [height, width]), class_count)
            Q1, S1, A1, superpixel_scount = ls.simple_superpixel(scale=superpixel_scale)  ### 超像素分割
            # Q 输入图和超像素之间的关系
            # S 初始化A的时候使用
            # A 邻接矩阵

            image_T1 = image_T1.reshape(-1, band)
            image_T2 = image_T2.reshape(-1, band)
            patch_T1 = np.dot(Q1.T, image_T1) / np.sum(Q1, axis=0).reshape(Q1.shape[1], 1).repeat(band, axis=1)
            patch_T2 = np.dot(Q1.T, image_T2) / np.sum(Q1, axis=0).reshape(Q1.shape[1], 1).repeat(band, axis=1)
            return Q1, S1, A1, superpixel_scount, patch_T1, patch_T2

        # Seed_List为随机种子点，curr_seed从1到5，这里的作用是每次生成一样的随机数
        for curr_seed in Seed_List:
            # step2:随机10%数据作为训练样本。方式：给出训练数据与测试数据的GT
            random.seed(curr_seed)  # 当seed()没有参数时，每次生成的随机数是不一样的，而当seed()有参数时，每次生成的随机数是一样的
            gt_reshape = np.reshape(gt, [-1])
            train_rand_idx = []
            if samples_type == 'ratio':  # 取一定比例训练
                for i in range(class_count):  # i从1跑到 class_count-1
                    idx = np.where(gt_reshape == i + 1)[-1]  # change
                    samplesCount = len(idx)
                    # print(samplesCount)
                    rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
                    rand_idx = random.sample(rand_list,
                                             np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
                    rand_real_idx_per_class = idx[rand_idx]
                    train_rand_idx.append(rand_real_idx_per_class)
                train_rand_idx = np.array(train_rand_idx, dtype=object)
                train_data_index = []
                for c in range(train_rand_idx.shape[0]):
                    a = train_rand_idx[c]
                    for j in range(a.shape[0]):
                        train_data_index.append(a[j])
                train_data_index = np.array(train_data_index, dtype=object)
                # 将测试集（所有样本，包括训练样本）也转化为特定形式
                train_data_index = set(train_data_index)  # label[0,1] 各取10% 存在一个二维数组里
                all_data_index = [i for i in range(len(gt_reshape))]
                all_data_index = set(all_data_index)  # 所有有label值的下标
                # 背景像元的标签
                background_idx = np.where(gt_reshape == 0)[-1]
                background_idx = set(background_idx)

                test_data_index = all_data_index - train_data_index - background_idx  # 测试的label 90%

                # 从测试集中随机选取部分样本作为验证集
                val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量 1%
                val_data_index = random.sample(list(test_data_index), val_data_count)
                val_data_index = set(val_data_index)
                test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集

                # 将训练集 验证集 测试集 整理
                test_data_index = list(test_data_index)
                train_data_index = list(train_data_index)
                val_data_index = list(val_data_index)

            # 获取训练样本的标签图
            train_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(train_data_index)):
                train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
                pass

            # 获取测试样本的标签图
            test_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(test_data_index)):
                test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
                pass

            Test_GT = np.reshape(test_samples_gt, [m, n])  # 测试样本图

            # 获取验证集样本的标签图
            val_samples_gt = np.zeros(gt_reshape.shape)
            for i in range(len(val_data_index)):
                val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
                pass

            train_samples_gt = np.reshape(train_samples_gt, [height, width])
            test_samples_gt = np.reshape(test_samples_gt, [height, width])
            val_samples_gt = np.reshape(val_samples_gt, [height, width])

            train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
            test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
            val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)

            train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
            test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
            val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)

            """
            制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量
            """


            # 训练集
            train_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            train_samples_gt = np.reshape(train_samples_gt, [m * n])
            for i in range(m * n):
                if train_samples_gt[i] != 0:
                    train_label_mask[i] = temp_ones
            train_label_mask = np.reshape(train_label_mask, [m * n, class_count])

            # 测试集
            test_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            test_samples_gt = np.reshape(test_samples_gt, [m * n])
            for i in range(m * n):
                if test_samples_gt[i] != 0:
                    test_label_mask[i] = temp_ones
            test_label_mask = np.reshape(test_label_mask, [m * n, class_count])

            # 验证集
            val_label_mask = np.zeros([m * n, class_count])
            temp_ones = np.ones([class_count])
            val_samples_gt = np.reshape(val_samples_gt, [m * n])
            for i in range(m * n):
                if val_samples_gt[i] != 0:
                    val_label_mask[i] = temp_ones
            val_label_mask = np.reshape(val_label_mask, [m * n, class_count])

            """
            change train_samples_gt
            """

            tic_all = time.time()
            tic0 = time.time()

            Q, S, A, superpixel_count, patch_T1, patch_T2 = image_into_patch(data1, data2, np.reshape(train_samples_gt,
                                                                                                      [height, width]),
                                                                             2, superpixel_scale)

            toc0 = time.time()
            LDA_SLIC_Time = toc0 - tic0

            print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))

            Q1 = torch.from_numpy(Q).to(device)
            A1 = torch.from_numpy(A).to(device)
            # 转到GPU
            train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
            test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
            val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
            # 转到GPU
            train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
            test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
            val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
            # 转到GPU
            train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
            test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
            val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

            net_input_1 = np.array(patch_T1, np.float32)

            net_input_2 = np.array(patch_T2, np.float32)


            def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor,
                             size: int):
                real_labels = reallabel_onehot
                we = -torch.mul(real_labels, torch.log(predict + 1e-45))
                we = torch.mul(we, reallabel_mask)
                pool_cross_entropy = torch.sum(we) / size
                return pool_cross_entropy



            zeros = torch.zeros([m * n]).to(device).float()

            def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                                     printFlag=True):
                if False == require_AA_KPP:
                    with torch.no_grad():
                        available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                        available_label_count = available_label_idx.sum()  # 有效标签的个数
                        correct_prediction = torch.where(
                            torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                            available_label_idx, zeros).sum()
                        OA = correct_prediction.cpu() / available_label_count

                        return OA
                else:
                    with torch.no_grad():
                        # 计算OA
                        available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                        available_label_count = available_label_idx.sum()  # 有效标签的个数

                        correct_prediction = torch.where(
                            torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),  # 如果最大值的位置一样就是1
                            available_label_idx, zeros).sum()
                        OA = correct_prediction.cpu() / available_label_count
                        OA = OA.cpu().numpy()

                        # 计算AA
                        zero_vector = np.zeros([class_count])
                        output_data = network_output.cpu().numpy()
                        train_samples_gt = train_samples_gt.cpu().numpy()
                        train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()

                        output_data = np.reshape(output_data, [m * n, class_count])
                        idx = np.argmax(output_data, axis=-1)
                        for z in range(output_data.shape[0]):
                            if ~(zero_vector == output_data[z]).all():
                                idx[z] += 1
                        # idx = idx + train_samples_gt
                        count_perclass = np.zeros([class_count])
                        correct_perclass = np.zeros([class_count])
                        for x in range(len(train_samples_gt)):
                            if train_samples_gt[x] != 0:
                                count_perclass[int(train_samples_gt[x] - 1)] += 1
                                if train_samples_gt[x] == idx[x]:
                                    correct_perclass[int(train_samples_gt[x] - 1)] += 1
                        test_AC_list = correct_perclass / count_perclass
                        test_AA = np.average(test_AC_list)

                        # 计算KPP
                        test_pre_label_list = []
                        test_real_label_list = []
                        output_data = np.reshape(output_data, [m * n, class_count])
                        idx = np.argmax(output_data, axis=-1)
                        idx = np.reshape(idx, [m, n])
                        for ii in range(m):
                            for jj in range(n):
                                if Test_GT[ii][jj] != 0:
                                    test_pre_label_list.append(idx[ii][jj] + 1)
                                    test_real_label_list.append(Test_GT[ii][jj])
                        test_pre_label_list = np.array(test_pre_label_list)
                        test_real_label_list = np.array(test_real_label_list)
                        kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                          test_real_label_list.astype(np.int16))
                        test_kpp = kappa

                        # 输出
                        if printFlag:
                            print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                            print('acc per class:')
                            print(test_AC_list)

                        OA_ALL.append(OA)
                        AA_ALL.append(test_AA)
                        KPP_ALL.append(test_kpp)
                        AVG_ALL.append(test_AC_list)

                        # 保存数据信息
                        f = open('results_' + dataset_name + '_results.txt', 'a+')
                        str_results = '\n======================' \
                                      + " learning rate=" + str(learning_rate) \
                                      + " epochs=" + str(max_epoch) \
                                      + " train ratio=" + str(train_ratio) \
                                      + " val ratio=" + str(val_ratio) \
                                      + " ======================" \
                                      + "\nOA=" + str(OA) \
                                      + "\nAA=" + str(test_AA) \
                                      + '\nkpp=' + str(test_kpp) \
                                      + '\nacc per class:' + str(test_AC_list) + "\n"
                        f.write(str_results)
                        f.close()
                        return OA


            data1 = torch.from_numpy(data1).permute(2, 0, 1).float()
            data2 = torch.from_numpy(data2).permute(2, 0, 1).float()
            data1 = torch.reshape(data1, (1, 224, 250, 250)).to(device)
            data2 = torch.reshape(data2, (1, 224, 250, 250)).to(device)


            tic_sam = time.time()

            def C_T(Q1, image_T1, sp):
                image_T1 = image_T1.permute(0, 2, 3, 1)
                _, height, width, band = image_T1.shape
                image_T1 = image_T1.reshape(-1, band)
                out = torch.mm(Q1.T / 1.0, image_T1) / torch.sum(Q1, dim=0).reshape(Q1.shape[1], 1).repeat(1, band)
                out = torch.reshape(out, (1, sp, band))
                return out

            data = torch.cat((data1, data2), 1)
            data_SLIC = C_T(Q1, data, superpixel_count).cpu()
            A = AdjMatrix(data_SLIC).to(device)
            toc_sam = time.time()
            print('sam_time:', toc_sam - tic_sam)

            print(Q1.shape, A.shape, superpixel_count)

            if dataset_name == "Bar":
                net = D_LIEG(Q1, A, superpixel_count, 224, 256, 3, 224, 8, 128, superpixel_count).to(
                    device)

            print("parameters", net.parameters(), len(list(net.parameters())))
            net.to(device)

            print(data1.shape)
            # 训练
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)  # weight_decay=0.0001
            best_loss = 999
            net.train()
            tic1 = time.time()
            gt = torch.from_numpy(gt_reshape).to(device) - torch.ones_like(torch.from_numpy(gt_reshape)).to(device)

            for i in range(max_epoch + 1):
                net.train()
                # print(gt.shape)

                output = net(data1, data2)
                # print(output.shape)
                loss1 = compute_loss(output, train_samples_gt_onehot, train_label_mask, len(
                    train_data_index))  # + torch.exp(1*( -0.5*sparseness(dots1)-0.5*sparseness(dots2)))
                loss = loss1
                print("epoch = {}".format(i), "train loss = {} ".format(
                    loss))  # ,"sparseness = {}".format(torch.exp(1*( -0.5*sparseness(dots1)-0.5*sparseness(dots2))))
                optimizer.zero_grad()  # zero the gradient buffers
                loss.backward()
                optimizer.step()  # Does the update
                if i % 1 == 0:
                    with torch.no_grad():
                        net.eval()
                        valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask, len(val_data_index))
                        if valloss < best_loss:
                            best_loss = valloss
                            torch.save(net.state_dict(), 'best_model.pt')
                            print('save model...')

            toc1 = time.time()

            toc_all = time.time()
            print('总时间：', toc_all - tic_all)
            print('train时间', toc1 - tic1)
            print('SLIC时间', toc0 - tic0)
            print("\n\n====================training done. starting evaluation...========================\n")
            training_time = toc1 - tic1 + LDA_SLIC_Time  # 分割耗时需要算进去
            Train_Time_ALL.append(training_time)

            torch.cuda.empty_cache()
            with torch.no_grad():

                net.load_state_dict(torch.load('best_model.pt'))
                net.eval()
                output = net(data1, data2)

                testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask,
                                        len(test_data_index))  # 0.5*sparseness(dots1)+0.5*sparseness(dots2)
                testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=False,
                                              printFlag=True)

                print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))

                output = output.argmax(dim=1)

                # lable = sio.loadmat(root + 'bay_gt.mat')
                # lable = lable[gt].reshape(250 * 250)
                # (OA, Kappa, OE, F1) = index(output, lable)

                output = output.reshape([height, width]).cpu().numpy()

                sio.savemat('Bay_1.mat', {'output': output})
                # 计算

                testing_time = toc_all - tic_all + LDA_SLIC_Time  # 分割耗时需要算进去
                Test_Time_ALL.append(testing_time)

            torch.cuda.empty_cache()
            del net

        OA_ALL = np.array(OA_ALL)
        AA_ALL = np.array(AA_ALL)
        KPP_ALL = np.array(KPP_ALL)
        AVG_ALL = np.array(AVG_ALL)
        Train_Time_ALL = np.array(Train_Time_ALL)
        print(Train_Time_ALL)
        print(toc1 - tic1)
        Test_Time_ALL = np.array(Test_Time_ALL)

        print("\ntrain_ratio={}".format(curr_train_ratio),
              "\n==============================================================================")
        print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
        print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
        print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
        print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
        print("Average training time:{}".format(np.mean(Train_Time_ALL)))
        print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

        # 保存数据信息
        f = open('results_' + dataset_name + '_results.txt', 'a+')
        str_results = '\n\n************************************************' \
                      + "\ntrain_ratio={}".format(curr_train_ratio) \
                      + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
                      + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
                      + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
                      + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
                      + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
                      + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
        f.write(str_results)
        f.close()

main()
