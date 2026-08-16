import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import API_KEY, BASE_URL, EMBEDDING_MODEL_NAME, MODEL_NAME
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

embeddings_model = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=EMBEDDING_MODEL_NAME,
    check_embedding_ctx_length=False
)

# Create embeddings from example sentences
text1 = "The cat sat on the mat"
text2 = "A feline rested on the carpet"
text3 = "Python is a programming language"

# Get embeddings using LangChain
embeddings = embeddings_model.embed_documents([text1, text2, text3])

# These similar sentences will have similar embeddings
embedding1 = embeddings[0]      # Embedding for "The cat sat on the mat"
embedding2 = embeddings[1]      # Embedding for "A feline rested on the carpet"
embedding3 = embeddings[2]      # Embedding for "Python is a programming language"

# Output shows number of documents and embedding dimensions
print(f"Number of documents: {len(embeddings)}")
print(f"Dimensions per embeddings: {len(embeddings[0])}")
