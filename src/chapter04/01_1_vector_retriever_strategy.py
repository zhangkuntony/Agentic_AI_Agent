import faiss
import numpy as np
import time

"""
具体向量索引策略的选择，可以参考教材第90页，图4.2——索引策略选择流程
"""

# 创建样例数据：10,000个向量，每个向量128维
dimension = 1024
num_vectors = 10000
generator = np.random.default_rng(42)
vectors = generator.random((num_vectors, dimension)).astype(np.float32)
query = generator.random((1, dimension)).astype(np.float32)
# 精确搜索索引
exact_index = faiss.IndexFlatL2(dimension)
exact_index.add(vectors)

# HNSW索引（近似但速度更快）
hnsw_index = faiss.IndexHNSWFlat(dimension, 32)         # 每个节点32个连接
hnsw_index.add(vectors)

# 比较搜索时间
start_time = time.time()
exact_D, exact_I = exact_index.search(query, k=10)          # 搜索10个最近邻
exact_time = time.time() - start_time

start_time = time.time()
hnsw_D, hnsw_I = hnsw_index.search(query, k=10)
hnsw_time = time.time() - start_time

# 计算重叠度（即两种方法返回相同结果的数量）
overlap = len(set(exact_I[0]).intersection(set(hnsw_I[0])))
overlap_percentage = overlap * 100 / 10

print(f"Exact search time: {exact_time:.6f} seconds")
print(f"HNSW search time: {hnsw_time:.6f} seconds")
print(f"Speed improvement: {exact_time/hnsw_time:.2f}x faster")
print(f"Result overlap: {overlap_percentage:.1f}%")
