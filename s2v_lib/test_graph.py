import torch

# 创建一个稀疏张量
# 索引张量指定非零元素的位置
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])  # 2x3的索引表示张量的非零值位置 第一行是 行索引，第二行是 列索引
# 值张量指定每个非零元素的值
values = torch.tensor([3, 4, 5], dtype=torch.float32)
# 指定稀疏张量的形状
size = (2, 3)

# 使用 indices 和 values 创建稀疏张量
sparse_tensor = torch.sparse.FloatTensor(indices, values, torch.Size(size))

print(sparse_tensor)
