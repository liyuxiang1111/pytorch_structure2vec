#include "s2v_lib.h"
#include "config.h"
#include "msg_pass.h"
#include "graph_struct.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <signal.h>

// 初始化函数，加载程序参数
int Init(const int argc, const char **argv)
{
    cfg::LoadParams(argc, argv);
    return 0;
}

// 创建并返回一个新的 GraphStruct 对象
void *GetGraphStruct()
{
    auto *batch_graph = new GraphStruct();
    return batch_graph;
}

// 准备批量图结构
// _batch_graph: 图结构指针
// num_graphs: 图的数量
// num_nodes: 每个图的节点数量数组
// num_edges: 每个图的边数量数组
// list_of_edge_pairs: 每个图的边对列表
// is_directed: 是否为有向图
int PrepareBatchGraph(void *_batch_graph,
                      const int num_graphs,
                      const int *num_nodes,
                      const int *num_edges,
                      void **list_of_edge_pairs,
                      int is_directed)
{
    GraphStruct *batch_graph = static_cast<GraphStruct *>(_batch_graph); // 转换图结构指针
    std::vector<unsigned> prefix_sum; // 存储每个图的节点前缀和
    prefix_sum.clear();
    unsigned edge_cnt = 0, node_cnt = 0;

    // 计算节点总数、边总数和节点前缀和
    for (int i = 0; i < num_graphs; ++i)
    {
        node_cnt += num_nodes[i];
        edge_cnt += num_edges[i];
        prefix_sum.push_back(num_nodes[i]);
        if (i)
            prefix_sum[i] += prefix_sum[i - 1];
    }
    for (int i = (int)prefix_sum.size() - 1; i > 0; --i)
        prefix_sum[i] = prefix_sum[i - 1]; // 将前缀和向右偏移一位
    prefix_sum[0] = 0;

    batch_graph->Resize(num_graphs, node_cnt); // 调整图结构大小

    // 为每个图的每个节点添加节点信息
    for (int i = 0; i < num_graphs; ++i)
    {
        for (int j = 0; j < num_nodes[i]; ++j)
        {
            batch_graph->AddNode(i, prefix_sum[i] + j);
        }
    }

    // 为每个图的每条边添加边信息
    int x, y, cur_edge = 0;
    for (int i = 0; i < num_graphs; ++i)
    {
        int *edge_pairs = static_cast<int *>(list_of_edge_pairs[i]);
        for (int j = 0; j < num_edges[i] * 2; j += 2)
        {
            x = prefix_sum[i] + edge_pairs[j];
            y = prefix_sum[i] + edge_pairs[j + 1];
            batch_graph->AddEdge(cur_edge, x, y);
            if (!is_directed)
                batch_graph->AddEdge(cur_edge + 1, y, x); // 如果是无向图，添加反向边
            cur_edge += 2;
        }
    }

    return 0;
}

// 计算图中的边对数量
// _graph: 图结构指针
int NumEdgePairs(void *_graph)
{
    GraphStruct *graph = static_cast<GraphStruct *>(_graph); // 转换图结构指针
    int cnt = 0;
    for (uint i = 0; i < graph->num_nodes; ++i)
    {
        auto in_cnt = graph->in_edges->head[i].size(); // 获取节点的入边数量
        cnt += in_cnt * (in_cnt - 1);  // 计算边对数量
	}
    return cnt;
}

// 准备用于 Mean Field 方法的图结构
// _batch_graph: 图结构指针
// list_of_idxes: 索引数组列表
// list_of_vals: 值数组列表
int PrepareMeanField(void *_batch_graph,
                     void **list_of_idxes,
                     void **list_of_vals)
{
    GraphStruct *batch_graph = static_cast<GraphStruct *>(_batch_graph); // 转换图结构指针
    n2n_construct(batch_graph,
                  static_cast<long long *>(list_of_idxes[0]),
                  static_cast<Dtype *>(list_of_vals[0])); // 节点到节点的构建
    e2n_construct(batch_graph,
                  static_cast<long long *>(list_of_idxes[1]),
                  static_cast<Dtype *>(list_of_vals[1])); // 边到节点的构建
    subg_construct(batch_graph,
                   static_cast<long long *>(list_of_idxes[2]),
                   static_cast<Dtype *>(list_of_vals[2])); // 子图的构建

    return 0;
}

// 准备用于 Loopy Belief Propagation 方法的图结构
// _batch_graph: 图结构指针
// list_of_idxes: 索引数组列表
// list_of_vals: 值数组列表
int PrepareLoopyBP(void *_batch_graph,
                   void **list_of_idxes,
                   void **list_of_vals)
{
    GraphStruct *batch_graph = static_cast<GraphStruct *>(_batch_graph); // 转换图结构指针
    n2e_construct(batch_graph,
                  static_cast<long long *>(list_of_idxes[0]),
                  static_cast<Dtype *>(list_of_vals[0])); // 节点到边的构建
    e2e_construct(batch_graph,
                  static_cast<long long *>(list_of_idxes[1]),
                  static_cast<Dtype *>(list_of_vals[1])); // 边到边的构建
    e2n_construct(batch_graph,
                  static_cast<long long *>(list_of_idxes[2]),
                  static_cast<Dtype *>(list_of_vals[2])); // 边到节点的构建
    subg_construct(batch_graph,
                   static_cast<long long *>(list_of_idxes[3]),
                   static_cast<Dtype *>(list_of_vals[3])); // 子图的构建
    return 0;                   
}