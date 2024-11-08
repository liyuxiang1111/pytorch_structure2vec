from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from pytorch_util import weights_init, to_scalar

'''
用与做回归任务的感知机
'''
class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()
        # 定义第一个全连接层，从输入层到隐藏层
        self.h1_weights = nn.Linear(input_size, hidden_size)
        # 定义第二个全连接层，从隐藏层到输出层，输出一个值用于回归
        self.h2_weights = nn.Linear(hidden_size, 1)
        # 初始化权重
        weights_init(self)

    def forward(self, x, y = None):
        # 前向传播：计算隐藏层的输出
        h1 = self.h1_weights(x)
        h1 = F.relu(h1) # 应用 ReLU 激活函数

        # 计算最终的预测值
        pred = self.h2_weights(h1)

        # 如果给定真实值 y，则计算损失
        if y is not None:
            y = Variable(y) # 将 y 转换为 PyTorch 变量
            mse = F.mse_loss(pred, y) # 计算均方误差 (MSE)
            mae = F.l1_loss(pred, y) # 计算平均绝对误差 (MAE)
            return pred, mae, mse
        else:
            return pred

'''
用与做分类任务的感知机
'''
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MLPClassifier, self).__init__()
        # 定义第一个全连接层，从输入层到隐藏层
        self.h1_weights = nn.Linear(input_size, hidden_size)
        # 定义第二个全连接层，从隐藏层到输出层，输出类别数量个值
        self.h2_weights = nn.Linear(hidden_size, num_class)
        # 初始化权重
        weights_init(self)

    def forward(self, x, y = None):
        # 前向传播：计算隐藏层的输出
        h1 = self.h1_weights(x)
        h1 = F.relu(h1) # 应用 ReLU 激活函数

        # 应用 ReLU 激活函数
        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1) # 计算每个类别的对数概率

        # 如果给定真实类别标签 y，则计算损失和准确率
        if y is not None:
            y = Variable(y) # 将 y 转换为 PyTorch 变量
            loss = F.nll_loss(logits, y) # 计算负对数似然损失

            # 预测类别
            pred = logits.data.max(1, keepdim=True)[1]

            # 计算准确率
            acc = to_scalar(pred.eq(y.data.view_as(pred)).sum())
            acc = float(acc) / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits
