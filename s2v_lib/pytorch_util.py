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

from s2v_lib import S2VLIB

# 检查张量是否是CUDA Float类型
def is_cuda_float(mat):
    version = get_torch_version()
    if version >= 0.4:
        return mat.is_cuda
    return type(mat) is torch.cuda.FloatTensor

# 将张量转换为标量
def to_scalar(mat):
    version = get_torch_version()
    if version >= 0.4:
        return mat.item() # PyTorch 0.4及以后可以直接使用item()获取标量
    return mat.data.cpu().numpy()[0]  # 之前版本需手动获取第一个元素

# 获取PyTorch版本号
def get_torch_version():
    return float('.'.join(torch.__version__.split('.')[0:2]))

# 使用Glorot均匀初始化方法（Xavier初始化）来初始化权重
def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size() # 全连接层的输入和输出通道数
    elif len(t.size()) == 3:
        # 卷积层，获取输入通道和输出通道数以及卷积核大小
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size()) # 高维权重张量的输入节点数量
        fan_out = np.prod(t.size()) # 高维权重张量的输出节点数量

    # 计算初始化范围
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit) # 使用均匀分布在范围内进行初始化

# 初始化模型的参数
def _param_init(m):
    if isinstance(m, Parameter): # 如果是Parameter类型，使用Glorot均匀初始化
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear): # 线性参数重置
        m.bias.data.zero_() # 将偏置置为0
        glorot_uniform(m.weight.data) # 初始化权重

# 为模型m进行权重初始化
def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList): # 确保代码能灵活处理包含多个参数列表的复杂网络结构
            for pp in p:
                _param_init(pp) # 初始化ParameterList中的每个参数
        else:
            _param_init(p) # 初始化其他参数

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters # 顶层参数
            _param_init(p)

# 自定义稀疏矩阵与密集矩阵的乘法操作
'''
使用 @staticmethod 装饰 forward 和 backward 方法的原因是：
保持 Function 类的无状态性，符合自动微分系统的要求。
使用 ctx 对象在前向和反向传播间传递数据，避免依赖实例属性。
提供便捷的调用方式，使得 forward 和 backward 的定义更加简洁。
'''
class MySpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        # 保存计算图中需要的张量。使用 ctx.save_for_backward 存储 forward 计算中的输入，供 backward 使用
        ctx.save_for_backward(sp_mat, dense_mat)
        # 返回稀疏矩阵与密集矩阵的乘积
        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):
        # 恢复前向计算中保存的张量
        sp_mat, dense_mat = ctx.saved_variables
        grad_matrix1 = grad_matrix2 = None

        # 不需要对sp_mat求梯度
        assert not ctx.needs_input_grad[0] # ctx.needs_input_grad[0] 表示 sp_mat 是否需要梯度。ctx.needs_input_grad[1] 表示 dense_mat 是否需要梯度。
        if ctx.needs_input_grad[1]: # 需要对dense_mat求梯度时计算梯度
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data)) # 以支持自动求导 返回为tensor
        
        return grad_matrix1, grad_matrix2 # 返回两个张量的梯度

# 包装MySpMM函数，用于执行稀疏矩阵与密集矩阵的乘法
def gnn_spmm(sp_mat, dense_mat):
    return MySpMM.apply(sp_mat, dense_mat) # apply 是 PyTorch 自动微分系统的一部分，它会记录前向传播过程，以便生成计算图。
