from langchain_community.document_loaders import JSONLoader
from langchain_community.retrievers import KNNRetriever
from langchain_community.retrievers.pubmed import PubMedRetriever
from xmltodict import parse
from langchain_community.vectorstores import FAISS
from utils.ark_embeddings import ArkMultimodalEmbeddings

# Basic RAG Implementation

# 1. Load documents
loader = JSONLoader(
    file_path="knowledge_base.json",
    jq_schema=".[].content",        # This extracts the content field from each array item
    text_content=True
)
documents = loader.load()

# 2. Convert to vectors
embedding_model = ArkMultimodalEmbeddings()
embeddings = embedding_model.embed_documents([doc.page_content for doc in documents])

# 3. Store in vector database
vector_db = FAISS.from_documents(documents, embedding_model)

# 4. Retrieve similar docs
query = "What are the effects of climate change?"
results = vector_db.similarity_search(query)
print(results)


# KNN Retriever
retriever = KNNRetriever.from_documents(documents, embedding_model)
results = retriever.invoke("query")
print(results)


# External Search API Retriever
retriever = PubMedRetriever(parse=parse)
results = retriever.invoke("COVID research")
print(results)