import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 添加s2v_lib路径
sys.path.append('%s/../s2v_lib' % os.path.dirname(os.path.realpath(__file__)))

# 导入S2V图嵌入方法和其他辅助工具
from embedding import EmbedMeanField, EmbedLoopyBP
from pytorch_util import to_scalar
from mlp import MLPClassifier
from util import cmd_args, load_data

# 定义分类器模型
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # 根据命令行参数选择图嵌入方法
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        # 定义图嵌入层（S2V模型）
        self.s2v = model(latent_dim=cmd_args.latent_dim, 
                        output_dim=cmd_args.out_dim,
                        num_node_feats=cmd_args.feat_dim, 
                        num_edge_feats=0, # 无边特征
                        max_lv=cmd_args.max_lv) # 消息传递的最大轮数

        # 确定输出层维度
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = cmd_args.latent_dim

        # 定义多层感知机（MLP）分类器
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class)

    # 准备节点特征和标签
    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph)) # 存储标签
        n_nodes = 0 # 节点总数
        concat_feat = [] # 存储所有节点的特征
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label # 获取图的标签
            n_nodes += batch_graph[i].num_nodes # 累加节点数
            concat_feat += batch_graph[i].node_tags # 收集节点特征

        # 将节点特征转换为长整型Tensor
        concat_feat = torch.LongTensor(concat_feat).view(-1, 1)
        node_feat = torch.zeros(n_nodes, cmd_args.feat_dim) # 初始化节点特征矩阵
        node_feat.scatter_(1, concat_feat, 1) # 将节点特征设置为1

        # 如果使用GPU，将数据移动到GPU
        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda() 
            labels = labels.cuda()

        return node_feat, labels

    # 前向传播过程
    def forward(self, batch_graph):
        # 准备节点特征和标签
        node_feat, labels = self.PrepareFeatureLabel(batch_graph)

        # 获取图的嵌入表示
        embed = self.s2v(batch_graph, node_feat, None)

        # 使用MLP进行分类
        return self.mlp(embed, labels)

# 定义训练/测试过程的循环
def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    # 存储每个batch的损失和准确率
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize # 计算迭代次数
    pbar = tqdm(range(total_iters), unit='batch')  # 用tqdm显示训练进度

    n_samples = 0 # 统计样本总数
    for pos in pbar:
        # 获取当前batch的图索引
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        # 根据索引获取图数据
        batch_graph = [g_list[idx] for idx in selected_idx]

        # 进行前向传播，计算损失和准确率
        _, loss, acc = classifier(batch_graph)

        # 如果是训练模式，执行反向传播和优化
        if optimizer is not None:
            optimizer.zero_grad() # 清零梯度
            loss.backward()       # 反向传播
            optimizer.step()      # 更新参数

        loss = to_scalar(loss) # 转换为标量
        # 更新进度条的描述信息
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )

        # 将当前batch的损失和准确率添加到total_loss
        total_loss.append( np.array([loss, acc]) * len(selected_idx))

        # 累计样本数量
        n_samples += len(selected_idx)

    if optimizer is None:
        assert n_samples == len(sample_idxes)

    # 计算平均损失和准确率
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss

# 主函数
if __name__ == '__main__':
    # 设置随机种子，确保实验可重复
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # 加载训练集和测试集图数据
    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    # 初始化分类器
    classifier = Classifier()

    # 如果使用GPU，将模型移动到GPU
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    # 使用Adam优化器
    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    # 训练集的索引列表
    train_idxes = list(range(len(train_graphs)))
    best_loss = None # 初始化最好的损失值

    # 训练循环
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes) # 随机打乱训练数据
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer) # 训练一个epoch
        # 打印训练损失和准确率
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1]))

        # 计算测试集的损失和准确率
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs)))) # 测试阶段不需要梯度更新
        # 打印测试损失和准确率
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1]))

        # 保存最佳模型（可选代码段）
        # if best_loss is None or test_loss[0] < best_loss:
        #     best_loss = test_loss[0]
        #     print('----saving to best model since this is the best valid loss so far.----')
        #     torch.save(classifier.state_dict(), cmd_args.save_dir + '/epoch-best.model')
        #     save_args(cmd_args.save_dir + '/epoch-best-args.pkl', cmd_args)
