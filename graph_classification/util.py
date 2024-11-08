from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
import networkx as nx

import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu') # 计算模式：cpu或gpu
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp') # 消息传递方式：mean_field 或 loopy_bp
cmd_opt.add_argument('-data', default=None, help='data folder name') # 数据文件夹名称
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size') # 批次大小
cmd_opt.add_argument('-seed', type=int, default=1, help='seed') # 随机种子
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of node feature') # 节点特征的维度
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes') # 类别数
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)') # 交叉验证的fold编号
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs') # 训练的epoch数
cmd_opt.add_argument('-latent_dim', type=int, default=64, help='dimension of latent layers') # 潜在层的维度
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='s2v output size') # S2V输出维度
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of regression') # 回归的维度
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing') # 最大消息传递轮数
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate') # 初始学习率

# 解析命令行参数
cmd_args, _ = cmd_opt.parse_known_args()

# 打印命令行参数
print(cmd_args)

# 定义一个图类S2VGraph，用于存储每个图的信息
class S2VGraph(object):
    def __init__(self, g, node_tags, label):
        self.num_nodes = len(node_tags) # 节点数
        self.node_tags = node_tags # 节点标签
        self.label = label # 图的标签

        # 获取图的边信息
        x, y = zip(*g.edges())
        self.num_edges = len(x)  # 边数
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x # 边的起点
        self.edge_pairs[:, 1] = y # 边的终点
        self.edge_pairs = self.edge_pairs.flatten() # 展平边信息

# 加载数据的函数
def load_data():
    print('loading data')

    # 存储图的列表、标签字典和特征字典
    g_list = []
    label_dict = {}
    feat_dict = {}

    # 读取数据文件
    with open('./data/%s/%s.txt' % (cmd_args.data, cmd_args.data), 'r') as f:
        n_g = int(f.readline().strip()) # 读取图的数量
        for i in range(n_g):
            row = f.readline().strip().split() # 读取每个图的节点数和标签
            n, l = [int(w) for w in row]
            # 如果标签尚未在字典中，添加标签映射
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            # 创建一个空的图
            g = nx.Graph()
            node_tags = [] # 用于存储每个节点的标签
            n_edges = 0 # 用于统计边的数量
            for j in range(n):
                g.add_node(j) # 添加节点
                row = f.readline().strip().split() # 读取节点特征
                row = [int(w) for w in row]
                # 如果特征尚未在字典中，添加特征映射
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                # 将节点的特征标签添加到节点标签列表
                node_tags.append(feat_dict[row[0]])

                n_edges += row[1] # 增加边数
                # 读取并添加节点之间的边
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])
            # 检查边的数量是否正确
            assert len(g.edges()) * 2 == n_edges
            # 检查节点数量是否正确
            assert len(g) == n
            # 将图及其标签和节点标签添加到图列表中
            g_list.append(S2VGraph(g, node_tags, l))

    # 对所有图的标签进行编号
    for g in g_list:
        g.label = label_dict[g.label]

    # 设置类数和节点特征维度
    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict)

    # 打印类别数和节点特征数
    print('# classes: %d' % cmd_args.num_class)
    print('# node features: %d' % cmd_args.feat_dim)

    # 读取训练集和测试集的索引
    train_data = np.loadtxt('./data/%s/10fold_idx/train_idx-%d.txt' % (cmd_args.data, cmd_args.fold), dtype=np.int32)
    test_data = np.loadtxt('./data/%s/10fold_idx/test_idx-%d.txt' % (cmd_args.data, cmd_args.fold), dtype=np.int32)
    train_idxes = train_data.tolist() # 训练集索引
    test_idxes = test_data.tolist() # 测试集索引

    # 处理训练集和测试集的维度
    if train_data.ndim == 0:
        train_idxes = [train_idxes]
    if test_data.ndim == 0:
        test_idxes = [test_idxes]

    # 返回训练集和测试集的图数据
    return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes]
    
