from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import ChatOpenAI
from utils.ark_embeddings import ArkMultimodalEmbeddings

chat_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

store = LocalFileStore("./cache/")

underlying_embeddings = ArkMultimodalEmbeddings()

# Avoiding unnecessary costs by caching the embeddings.
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)