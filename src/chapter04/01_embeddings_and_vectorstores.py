from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.ark_embeddings import ArkMultimodalEmbeddings

# Initialize the embeddings model with Ark Embeddings Model
embeddings_model = ArkMultimodalEmbeddings()

# # 1. Basic Embeddings Usage
#
# # Create embeddings from example sentences
# text1 = "The cat sat on the mat"
# text2 = "A feline rested on the carpet"
# text3 = "Python is a programming language"
#
# # Get embeddings using LangChain
# embeddings = embeddings_model.embed_documents([text1, text2, text3])
#
# # These similar sentences will have similar embeddings
# embedding1 = embeddings[0]      # Embedding for "The cat sat on the mat"
# embedding2 = embeddings[1]      # Embedding for "A feline rested on the carpet"
# embedding3 = embeddings[2]      # Embedding for "Python is a programming language"
#
# # Output shows number of documents and embedding dimensions
# print(f"Number of documents: {len(embeddings)}")
# print(f"Dimensions per embeddings: {len(embeddings[0])}")
# print(f"Sample of embeddings: {embeddings[0][:10]}")

# 2. Vector Store Setup

# Create some sample documents with explicit IDs
docs = [
    Document(page_content="Content about language models", metadata={"id": "doc_1"}),
    Document(page_content="Information about vector databases", metadata={"id": "doc_2"}),
    Document(page_content="Details about retrieval systems", metadata={"id": "doc_3"})
]

# Create the vector store
vector_store = Chroma(embedding_function=embeddings_model)

# Add documents with explicit IDs
vector_store.add_documents(docs)

# Similarity Search with appropriate k value
results = vector_store.similarity_search("How do language models work?", k=2)
print(results)

# For MMR, adjust the parameter based on available documents
found_docs = vector_store.similarity_search("retrieval", k=1)
print(f"Found documents: {len(found_docs)}")

# Identify with documents were already retrieved
used_doc_ids = {doc.metadata["id"] for doc in found_docs}

# Build a list of remaining (unused) documents
remaining_docs = [doc for doc in docs if doc.metadata["id"] not in used_doc_ids]
print(f"Remaining documents available for MMR: {len(remaining_docs)}")

"""
MMR 全称是 Maximal Marginal Relevance（最大边际相关性），是信息检索里一种用于**平衡「相关性」和「多样性」**的排序算法。

核心思想
在普通相似度搜索里，返回的结果往往只是「最相似的一堆」，可能彼此内容高度重复。MMR 的目标是：既要结果和查询相关，又要结果之间尽量不重复，让返回的文档更「多样、互补」。

它通过一个公式，在选下一篇文章时，同时考虑两个因素：

相关性（Relevance）：这篇文章和查询 query 有多相关；
冗余度（Redundancy）：这篇文章和「已经选出来的文章」有多相似（越相似越冗余，越该被扣分）。

lambda_mult 的作用
lambda_mult = 1：只追求相关性，退化成普通的相似度搜索，不管多样性；
lambda_mult = 0：只追求多样性，结果之间尽量不同，但不关心和查询相关；
默认常见取 0.5：相关性和多样性各占一半。

这里 k=1, fetch_k=1，候选池只有 1 条，MMR 的「多样性」根本没机会发挥作用（没有多余候选可选）。
这是因为前面只往库里加了 3 篇文档、且 found_docs 已取走部分，为了避免报「k 超过可用数量」才把值压到 1。

如果想真正体验 MMR 的「多样性」效果，需要满足：fetch_k > k，且向量库里有足够多的文档。当前这个 demo 文档太少，MMR 退化成近似普通搜索，是正常的。
"""

if len(remaining_docs) > 0:
    mmr_results = vector_store.max_marginal_relevance_search(
        "retrieval systems",
        k=1,        # Only request what's available
        fetch_k=1,  # Only fetch what's available
        lambda_mult=0.5
    )
    print(f"MMR results: {mmr_results}")
