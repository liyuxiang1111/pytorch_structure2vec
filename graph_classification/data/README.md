### txt data format

* 1st line: `N` number of graphs; then the following `N` blocks describe the graphs  
* for each block of text:
  - a line contains `n l`, where `n` is number of nodes in the current graph, and `l` is the graph label
  - following `n` lines: 
    - the `i`th line describes the information of `i`th node (0 based), which starts with `t m`, where `t` is the tag of current node, and `m` is the number of neighbors of current node;
    - following `m` numbers indicate the neighbor indices (starting from 0). 
    
### mat data format

It should be straightforward when you load it with Matlab.

### 数据介绍
#### 图分类数据集介绍

这些数据集都是用于图分类任务的标准数据集，通常用于测试图神经网络模型的性能。以下是每个数据集的具体介绍：

##### 1. DD
DD 是一个生物学领域的图数据集，其中的每个图代表一组蛋白质结构。具体内容如下：

- **节点**：氨基酸残基
- **边**：残基之间的关系
- **任务**：预测不同蛋白质的功能分类

##### 2. ENZYMES
ENZYMES 是一个生物化学领域的数据集，用于分类酶的种类。其详细信息如下：

- **节点**：酶分子结构中的原子
- **边**：原子之间的连接
- **任务**：将酶分为六种功能类别之一

##### 3. MUTAG
MUTAG 是一个化学分子数据集，包含了用于毒性预测的分子结构。详细信息如下：

- **节点**：化学分子中的原子
- **边**：原子之间的化学键
- **任务**：预测分子是否具有突变原性（即是否可能引发基因突变）

##### 4. NCI1
NCI1 是从美国国家癌症研究所（NCI）获得的一个化学分子数据集，包含多种不同的化合物。其详细信息如下：

- **节点**：化学化合物的原子
- **边**：原子之间的连接
- **任务**：预测化合物是否具有抗癌活性

##### 5. NCI109
NCI109 数据集与 NCI1 类似，也来自 NCI。它包含不同的分子结构，标签表示化合物是否对特定的癌细胞具有活性。其详细信息如下：

- **节点**：化学化合物的原子
- **边**：原子之间的连接
- **任务**：预测化合物是否对特定癌细胞有活性

这些数据集通常包含节点、边的属性以及图的标签，适用于评估和对比不同图分类算法的效果。
