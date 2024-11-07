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

from s2v_lib import S2VLIB # 导入自定义的结构化数据处理库
from pytorch_util import weights_init, gnn_spmm, is_cuda_float # 导入自定义的辅助函数

# EmbedMeanField 类：实现基于均值场（Mean Field）的图神经网络嵌入
class EmbedMeanField(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv = 3):
        """
        初始化嵌入模型，定义层级结构和参数。
        latent_dim: 潜在层的维度（隐层大小）
        output_dim: 输出维度（如果有的话）
        num_node_feats: 节点特征的数量
        num_edge_feats: 边特征的数量
        max_lv: 最大的层数，控制迭代的次数
        """
        super(EmbedMeanField, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.max_lv = max_lv

        # 定义各个变换层
        self.w_n2l = nn.Linear(num_node_feats, latent_dim) # 节点特征到潜在空间的线性变换
        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim) # 边特征到潜在空间的线性变换
        if output_dim > 0:
            self.out_params = nn.Linear(latent_dim, output_dim) # 输出层
        self.conv_params = nn.Linear(latent_dim, latent_dim) # 图卷积层
        weights_init(self) # 初始化权重

    def forward(self, graph_list, node_feat, edge_feat):
        """
        前向传播，计算图嵌入。
        graph_list: 图的列表
        node_feat: 节点特征
        edge_feat: 边特征
        """
        # 使用S2VLIB中的PrepareMeanField函数准备稀疏矩阵
        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)

        # 如果是CUDA上的浮点数据，转移到GPU
        if is_cuda_float(node_feat):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()


        node_feat = Variable(node_feat) # 将节点特征转化为Variable
        if edge_feat is not None:
            edge_feat = Variable(edge_feat) # 如果有边特征，将其转化为Variable

        # 将稀疏矩阵转化为Variable
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)

        # 调用mean_field函数计算最终的嵌入
        h = self.mean_field(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp)
        
        return h

    def mean_field(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp):
        """
        基于均值场算法的消息传递机制。
        node_feat: 节点特征
        edge_feat: 边特征
        n2n_sp, e2n_sp, subg_sp: 稀疏矩阵
        """
        # 节点特征经过线性层变换
        input_node_linear = self.w_n2l(node_feat)
        input_message = input_node_linear

        # 如果有边特征，进行边的线性变换并结合消息
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear) # 使用图的边特征与消息进行稀疏矩阵乘法
            input_message += e2npool_input

        # ReLU激活函数
        input_potential = F.relu(input_message)

        # 多层迭代过程
        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) # 稀疏矩阵乘法
            node_linear = self.conv_params( n2npool ) # 图卷积层
            merged_linear = node_linear + input_message # 合并节点消息

            cur_message_layer = F.relu(merged_linear)
            lv += 1

        # 如果有输出层，则通过输出层进行变换
        if self.output_dim > 0:
            out_linear = self.out_params(cur_message_layer)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = cur_message_layer

        # 最终通过子图进行消息传递
        y_potential = gnn_spmm(subg_sp, reluact_fp)

        return F.relu(y_potential) # 返回结果的ReLU激活

class EmbedLoopyBP(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv = 3):
        """
        初始化嵌入模型，定义层级结构和参数。
        latent_dim: 潜在层的维度（隐层大小）
        output_dim: 输出维度（如果有的话）
        num_node_feats: 节点特征的数量
        num_edge_feats: 边特征的数量
        max_lv: 最大的层数，控制迭代的次数
        """
        super(EmbedLoopyBP, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.max_lv = max_lv

        # 定义各个变换层
        self.w_n2l = nn.Linear(num_node_feats, latent_dim) # 节点特征到潜在空间的线性变换
        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim) # 边特征到潜在空间的线性变换
        if output_dim > 0:
            self.out_params = nn.Linear(latent_dim, output_dim) # 输出层
        self.conv_params = nn.Linear(latent_dim, latent_dim) # 图卷积层
        weights_init(self) # 初始化权重

    def forward(self, graph_list, node_feat, edge_feat):
        """
        前向传播，计算图嵌入。
        graph_list: 图的列表
        node_feat: 节点特征
        edge_feat: 边特征
        """
        # 使用S2VLIB中的PrepareLoopyBP函数准备稀疏矩阵
        n2e_sp, e2e_sp, e2n_sp, subg_sp = S2VLIB.PrepareLoopyBP(graph_list)

        # 如果是CUDA上的浮点数据，转移到GPU
        if is_cuda_float(node_feat):
            n2e_sp = n2e_sp.cuda()
            e2e_sp = e2e_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()

        node_feat = Variable(node_feat) # 将节点特征转化为Variable
        if edge_feat is not None:
            edge_feat = Variable(edge_feat) # 如果有边特征，将其转化为Variable

        # 将稀疏矩阵转化为Variable
        n2e_sp = Variable(n2e_sp)
        e2e_sp = Variable(e2e_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)

        # 调用loopy_bp函数计算最终的嵌入
        h = self.loopy_bp(node_feat, edge_feat, n2e_sp, e2e_sp, e2n_sp, subg_sp)
        
        return h

    def loopy_bp(self, node_feat, edge_feat, n2e_sp, e2e_sp, e2n_sp, subg_sp):
        """
        基于Loopy Belief Propagation (Loopy BP) 的消息传递机制。
        node_feat: 节点特征
        edge_feat: 边特征
        n2e_sp, e2e_sp, e2n_sp, subg_sp: 稀疏矩阵
        """
        # 节点特征经过线性层变换
        input_node_linear = self.w_n2l(node_feat)
        n2epool_input = gnn_spmm(n2e_sp, input_node_linear) # 稀疏矩阵乘法
        input_message = n2epool_input

        # 如果有边特征，进行边的线性变换并结合消息
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            input_message += input_edge_linear

        # ReLU激活函数
        input_potential = F.relu(input_message)

        # 多层迭代过程
        lv = 0
        cur_message_layer = input_potential
        while lv < self.max_lv:
            e2epool = gnn_spmm(e2e_sp, cur_message_layer) # 多层迭代过程
            edge_linear = self.conv_params(e2epool) # 图卷积层
            merged_linear = edge_linear + input_message # 合并边消息

            cur_message_layer = F.relu(merged_linear)
            lv += 1

        # 使用边特征与节点特征进行消息传递
        e2npool = gnn_spmm(e2n_sp, cur_message_layer)
        hidden_msg = F.relu(e2npool)

        # 如果有输出层，则通过输出层进行变换
        if self.output_dim > 0:
            out_linear = self.out_params(hidden_msg)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = hidden_msg

        # 最终通过子图进行消息传递
        y_potential = gnn_spmm(subg_sp, reluact_fp)

        return F.relu(y_potential) # 返回结果的ReLU激活
