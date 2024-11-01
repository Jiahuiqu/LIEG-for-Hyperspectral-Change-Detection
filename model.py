#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:39:32 2021

@author: xidian
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:49:43 2021

@author: xidian
"""
import torch
from einops import rearrange
from thop import profile
from torch import nn, einsum

#########获取图像的h,w
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Attention(nn.Module):
    def __init__(self, dim, A, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  ##Q,K,V的尺寸
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  ##乘以3是为了后面拆分成Q,K,V三部分

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.A = A

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  ##(b,121,103,head=8) [1,169,256,8]

        ##通过FC操作得到初始的Q,K,V
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # self.to_qkv(x)是(b,121,64*8*3)  qkv是(b,121,64*8)
        # chunk将tensor按dim方向分割成chunks个tensor块，返回的是一个元组。

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # [b, 8, 121, 64]  方便后面计算，也即是8个头单独计算

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # Q乘以K的转置再除以根号dim_head

        dots = dots + self.A

        attn = self.attend(dots)  # softmax操作
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # 使用einsum表示矩阵乘法：
        out = rearrange(out, 'b h n d -> b n (h d)')  # out:[b, 121, 512]
        return self.to_out(out), dots


####### Norm 模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


######### FeedForward 模块
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


###transformer模块attention+Norm  FeedForward + Norm
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, hidden_dim, A, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):  # depth决定有几个Transformer的encoder模块
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, A=A)),  ## attention+Norm
                PreNorm(dim, FeedForward(dim, hidden_dim, dropout=dropout))  ##FeedForward + Norm
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            a, dots = attn(x)
            x = a + x  ###第一个残差 attn=attention+Norm
            x = ff(x) + x  ###第二个残差 ff=FeedForward + Norm
        return x, dots


class Encoder(nn.Module):
    def __init__(self, *, image_size, num_patch, init_dim, dim, depth, heads, hidden_dim, A, pool='cls',
                 channels, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        num_patches = num_patch
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  # 169 256
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, hidden_dim, A, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()



    def forward(self, x):

        b, n, _ = x.shape

        x += self.pos_embedding[:, :(n)]  # x加上位置编码(b,N,d) ([1, 121, 103])
        x = self.dropout(x)

        x, dots = self.transformer(x)  ## Transformer的输入维度x的shape是：(b,N,d)

        return x.squeeze(0), dots


def C_T(Q1, image_T1, sp):
    image_T1 = image_T1.permute(0, 2, 3, 1)
    _, height, width, band = image_T1.shape
    image_T1 = image_T1.reshape(-1, band)
    out = torch.mm(Q1.T / 1.0, image_T1) / torch.sum(Q1, dim=0).reshape(Q1.shape[1], 1).repeat(1, band)
    out = torch.reshape(out, (1, sp, band))
    return out


class D_LIEG(nn.Module):
    def __init__(self, Q, A, num_patch, init_dim, dim, depth, channels, head, hidden_dim, superpixel):
        super(D_LIEG, self).__init__()
        self.transform1_1 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )
        self.transform2_1 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )

        self.transform1_2 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )
        self.transform2_2 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )

        self.transform1_3 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )
        self.transform2_3 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )
        self.transform1_4 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )
        self.transform2_4 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )
        self.transform1_5 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )
        self.transform2_5 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )

        self.transform1_6 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )
        self.transform2_6 = Encoder(image_size=5,  # 输入图片大小
                                    num_patch=num_patch,  ##将图片划分成patch 每个patch展平为向量
                                    init_dim=init_dim,
                                    dim=dim,  # 隐变量的维数(降维以后的维数)
                                    depth=depth,  # encoder的层数
                                    heads=head,  ##attention的head数
                                    hidden_dim=hidden_dim,  # FeedForward的隐藏层数  输出以后再reshape回原来的尺寸
                                    channels=channels,
                                    dropout=0.2,  # dropout rate
                                    emb_dropout=0.2,  # position Embedding dropout rate
                                    A=A
                                    )

        self.conv1 = nn.Conv2d(224, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(224, 256, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(224)
        self.bn2 = nn.BatchNorm2d(224)
        self.bn3 = nn.BatchNorm2d(224)
        self.bn4 = nn.BatchNorm2d(224)
        self.bn5 = nn.BatchNorm2d(224)
        self.bn6 = nn.BatchNorm2d(224)
        self.bn7 = nn.BatchNorm2d(224)
        self.bn8 = nn.BatchNorm2d(224)
        self.bn9 = nn.BatchNorm2d(224)
        self.bn10 = nn.BatchNorm2d(224)
        self.bn11 = nn.BatchNorm2d(224)
        self.bn12 = nn.BatchNorm2d(224)
        self.bn13 = nn.BatchNorm2d(224)
        self.bn14 = nn.BatchNorm2d(224)
        self.relu = nn.ReLU()

        self.fc0_1 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(224 + 256, 256))
        self.fc0_2 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(224 + 256, 256))
        self.fc1 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc2 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc3 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc4 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc5 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc6 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc7 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc8 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc9 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc10 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc11 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))
        self.fc12 = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(256 + 256, 256))


        self.Softmax_linear = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
        self.softmax = nn.Softmax(dim=1)
        self.Q = Q

        self.sp = superpixel

    def forward(self, T1, T2):
        # block1
        T1_LS = T1
        T2_LS = T2

        T1_LS = C_T(self.Q, T1_LS, self.sp)
        T2_LS = C_T(self.Q, T2_LS, self.sp)

        T1_out1 = self.conv1(T1)
        T2_out1 = self.conv2(T2)

        T1_out_tr1 = C_T(self.Q, T1_out1, self.sp)
        T2_out_tr1 = C_T(self.Q, T2_out1, self.sp)

        T1_out_tr1 = torch.reshape(T1_out_tr1, (1, self.sp, 256))
        T2_out_tr1 = torch.reshape(T2_out_tr1, (1, self.sp, 256))

        T1_out_tr1 = torch.cat((T1_out_tr1, T1_LS), 2)
        T2_out_tr1 = torch.cat((T2_out_tr1, T2_LS), 2)

        T1_out_tr1 = self.fc0_1(T1_out_tr1)
        T2_out_tr1 = self.fc0_2(T2_out_tr1)

        # print(T1_out_tr1.shape)
        T1_out_tr1, dots1 = self.transform1_1(T1_out_tr1)
        T2_out_tr1, dots2 = self.transform2_1(T2_out_tr1)
        # block2
        T1_out2 = self.conv3(T1_out1)
        T2_out2 = self.conv4(T2_out1)

        T1_out_tr2 = C_T(self.Q, T1_out2, self.sp)
        T2_out_tr2 = C_T(self.Q, T2_out2, self.sp)

        T1_out_tr2 = torch.reshape(T1_out_tr2, (self.sp, 256))
        T2_out_tr2 = torch.reshape(T2_out_tr2, (self.sp, 256))

        T1_out_tr2 = torch.cat((T1_out_tr2, T1_out_tr1), 1)
        T2_out_tr2 = torch.cat((T2_out_tr2, T2_out_tr1), 1)

        T1_out_tr2 = self.fc3(T1_out_tr2)
        T2_out_tr2 = self.fc4(T2_out_tr2)

        T1_out_tr2 = torch.reshape(T1_out_tr2, (1, self.sp, 256))
        T2_out_tr2 = torch.reshape(T2_out_tr2, (1, self.sp, 256))

        T1_out_tr2, dots1 = self.transform1_2(T1_out_tr2)
        T2_out_tr2, dots1 = self.transform2_2(T2_out_tr2)

        # block3
        T1_out3 = self.conv5(T1_out2)
        T2_out3 = self.conv6(T2_out2)

        T1_out_tr3 = C_T(self.Q, T1_out3, self.sp)
        T2_out_tr3 = C_T(self.Q, T2_out3, self.sp)

        T1_out_tr3 = torch.reshape(T1_out_tr3, (self.sp, 256))
        T2_out_tr3 = torch.reshape(T2_out_tr3, (self.sp, 256))

        T1_out_tr3  = torch.cat((T1_out_tr3, T1_out_tr2), 1)
        T2_out_tr3 = torch.cat((T2_out_tr3, T2_out_tr2), 1)

        T1_out_tr3 = self.fc5(T1_out_tr3)
        T2_out_tr3 = self.fc6(T2_out_tr3)

        T1_out_tr3 = torch.reshape(T1_out_tr3, (1, self.sp, 256))
        T2_out_tr3 = torch.reshape(T2_out_tr3, (1, self.sp, 256))

        T1_out_tr3, dots1 = self.transform1_3(T1_out_tr3)
        T2_out_tr3, dots1 = self.transform2_3(T2_out_tr3)

        # block4
        T1_out4 = self.conv7(T1_out3)
        T2_out4 = self.conv8(T2_out3)

        T1_out_tr4 = C_T(self.Q, T1_out4, self.sp)
        T2_out_tr4 = C_T(self.Q, T2_out4, self.sp)

        T1_out_tr4 = torch.reshape(T1_out_tr4, (self.sp, 256))
        T2_out_tr4 = torch.reshape(T2_out_tr4, (self.sp, 256))

        T1_out_tr4 = torch.cat((T1_out_tr4, T1_out_tr3), 1)
        T2_out_tr4 = torch.cat((T2_out_tr4, T2_out_tr3), 1)

        T1_out_tr4 = self.fc7(T1_out_tr4)
        T2_out_tr4 = self.fc8(T2_out_tr4)

        T1_out_tr4 = torch.reshape(T1_out_tr4, (1, self.sp, 256))
        T2_out_tr4 = torch.reshape(T2_out_tr4, (1, self.sp, 256))

        T1_out_tr4, dots1 = self.transform1_4(T1_out_tr4)
        T2_out_tr4, dots1 = self.transform2_4(T2_out_tr4)

        # block5
        T1_out5 = self.conv9(T1_out4)
        T2_out5 = self.conv10(T2_out4)

        T1_out_tr5 = C_T(self.Q, T1_out5, self.sp)
        T2_out_tr5 = C_T(self.Q, T2_out5, self.sp)

        T1_out_tr5 = torch.reshape(T1_out_tr5, (self.sp, 256))
        T2_out_tr5 = torch.reshape(T2_out_tr5, (self.sp, 256))

        T1_out_tr5 = torch.cat((T1_out_tr5, T1_out_tr4), 1)
        T2_out_tr5 = torch.cat((T2_out_tr5, T2_out_tr4), 1)

        T1_out_tr5 = self.fc9(T1_out_tr5)
        T2_out_tr5 = self.fc10(T2_out_tr5)

        T1_out_tr5 = torch.reshape(T1_out_tr5, (1, self.sp, 256))
        T2_out_tr5 = torch.reshape(T2_out_tr5, (1, self.sp, 256))

        T1_out_tr5, dots1 = self.transform1_5(T1_out_tr5)
        T2_out_tr5, dots1 = self.transform2_5(T2_out_tr5)


        # 输出
        T1_out6 = self.conv11(T1_out5)
        T2_out6 = self.conv12(T2_out5)

        T1_out_tr6 = C_T(self.Q, T1_out6, self.sp)
        T2_out_tr6 = C_T(self.Q, T2_out6, self.sp)

        T1_out_tr6 = torch.reshape(T1_out_tr6, (self.sp, 256))
        T2_out_tr6 = torch.reshape(T2_out_tr6, (self.sp, 256))

        T1_out_tr6 = torch.cat((T1_out_tr6, T1_out_tr5), 1)
        T2_out_tr6 = torch.cat((T2_out_tr6, T2_out_tr5), 1)

        T1_out = self.fc11(T1_out_tr6)
        T2_out = self.fc12(T2_out_tr6)

        out = T1_out - T2_out

        out = torch.matmul(self.Q.float(), out.float())
        out = self.softmax(self.Softmax_linear(out))
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    Q1 = torch.randint(0, 2, size=(62500, 335)).cuda()
    A = torch.rand(335, 335).cuda()
    img1 = torch.rand(1, 224, 250, 250).cuda()
    img2 = torch.rand(1, 224, 250, 250).cuda()
    A[:, :] = float('-inf')
    superpixel_count = 335
    model = CD_DI(Q1, A, superpixel_count, 224, 256, 3, 224, 8, 128, superpixel_count).cuda()
    flops, params = profile(model, inputs=(img1, img2))
    print(flops, params)

