class MolGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes  # 节点列表，包含每个节点的标签
        self.edges = edges  # 边列表，包含 (节点1, 节点2) 的元组

    def __repr__(self):
        return f"MolGraph(nodes={self.nodes}, edges={self.edges})"


# 创建简单的图数据
def create_sample_batch_graph():
    # 定义一些简单的图，每个图都有节点和边
    graph1 = MolGraph(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)])
    graph2 = MolGraph(nodes=[0, 1, 2, 3], edges=[(0, 1), (1, 2), (2, 3)])
    graph3 = MolGraph(nodes=[0, 1], edges=[(0, 1)])

    # 将这些图放入 batch_graph 列表中
    batch_graph = [graph1, graph2, graph3]
    return batch_graph


# 生成 batch_graph
batch_graph = create_sample_batch_graph()
print(batch_graph)
