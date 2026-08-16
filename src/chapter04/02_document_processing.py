from langchain_community.document_loaders import JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from utils.ark_embeddings import ArkMultimodalEmbeddings

# # 1. Document loaders
#
# # Load a json file
# loader = JSONLoader(
#     file_path="knowledge_base.json",
#     jq_schema=".[].content",        # This extracts the content field from each array item
#     text_content=True
# )
# documents = loader.load()
# print(documents)
#
# # 2. Fixed-Size chunking
#
# text_splitter = CharacterTextSplitter(
#     separator=" ",          # Split on spaces to avoid breaking words
#     chunk_size=200,
#     chunk_overlap=20
# )
#
# chunks = text_splitter.split_documents(documents)
# print(f"Generated {len(chunks)} chunks from document")
#
# for chunk in chunks:
#     print(chunk)


# 3. Recursive Character Chunking

"""
递归字符分块：
首先尝试在段落换行符（\n\n）处分割文本；若分割后的块仍较大，则尝试下一个分隔符（\n），依此类推。这种方法在保持合理分块大小的同时，尽量保留自然的文本边界。
对于大多数应用来说，递归字符分块是推荐的默认策略，适用于多种文档类型，并且在保留上下文与保持可管理分块大小之间提供了良好的平衡。
"""

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=150,
    chunk_overlap=20
)

document = """# Introduction to RAG
Retrieval-Augmented Generation (RAG) combines retrieval systems with generative AI models.

It helps address hallucinations by grounding responses in retrieved information.

## Key Components
RAG consists of several components:
1. Document processing
2. Vector embedding
3. Retrieval
4. Augmentation
5. Generation

### Document Processing
This step involves loading and chunking documents appropriately.
"""

chunks = text_splitter.split_text(document)
print(chunks)


# 4. Semantic Chunking

"""语义分块通过分析内容的含义来确定分块边界"""
embeddings = ArkMultimodalEmbeddings()
text_splitter = SemanticChunker(
    embeddings=embeddings,
    add_start_index=True            # Include position metadata
)

chunks = text_splitter.split_text(document)
print(chunks)
