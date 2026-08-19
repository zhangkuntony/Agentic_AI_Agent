from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from utils.ark_embeddings import ArkMultimodalEmbeddings

# # 1. Query Expansion
#
# expansion_template = """Given the user question: {question}
# Generate three alternative versions that express the same information need but with different wording:
# 1."""
#
# expansion_prompt = PromptTemplate(
#     input_variables=["question"],
#     template=expansion_template
# )
#
# llm = ChatOpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL,
#     model=MODEL_NAME,
#     temperature=0.7
# )
# expansion_chain = expansion_prompt | llm | StrOutputParser()
#
# # Generate expanded queries
# original_query = "What are the effects of climate change?"
# expanded_queries = expansion_chain.invoke(original_query)
# print(expanded_queries)
#
# 2. Hypothetical Document Embeddings (HyDE, 假设文档嵌入)

loader = JSONLoader(
    file_path="knowledge_base.json",
    jq_schema=".[].content",
    text_content=True
)
documents = loader.load()
embedder = ArkMultimodalEmbeddings()
embeddings = embedder.embed_documents([doc.page_content for doc in documents])
vector_db = FAISS.from_documents(documents, embedder)
#
# # Create prompt for generating hypothetical document
# hyde_template = """Based on the question: {question}
# Write a passage that could contain the answer to this question:"""
#
# hyde_prompt = PromptTemplate(
#     input_variables=["question"],
#     template=hyde_template
# )
# llm = ChatOpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL,
#     model=MODEL_NAME,
#     temperature=0.2
# )
# hyde_chain = hyde_prompt | llm | StrOutputParser()
#
# # Generate hypothetical document
# query = "What dietary changes can reduce carbon footprint?"
# hypothetical_doc = hyde_chain.invoke(query)
#
# # Use the hypothetical document for retrieval
# embeddings = ArkMultimodalEmbeddings()
# embedded_query = embeddings.embed_query(hypothetical_doc)
# results = vector_db.similarity_search_by_vector(embedded_query, k=3)
# print(results)
#
# # 3. Contextual Compression
#
# llm = ChatOpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL,
#     model=MODEL_NAME,
#     temperature=0
# )
# compressor = LLMChainExtractor.from_llm(llm)
#
# # Create a basic retriever from the vector store
# base_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
#
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=base_retriever,
# )
# compressed_doc = compression_retriever.invoke("How do transformers work?")
# print(compressed_doc)

# 4. Maximum Marginal Relevance (MMR, 最大边际相关性)

mmr_results = vector_db.max_marginal_relevance_search(
    query="What are transformer models?",
    k=5,
    fetch_k=20,
    lambda_mult=0.5     # Diversity parameter (0 = max diversity, 1 = max relevance)
)
print(mmr_results)