import os
import tempfile

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Any

from chapter04.chat_bot.document_loader import load_document
from chapter04.chat_bot.llms import EMBEDDINGS

VECTOR_STORE = InMemoryVectorStore(embedding=EMBEDDINGS)

def split_documents(docs: List[Document]) -> list[Document]:
    """Split each document."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200
    )
    return text_splitter.split_documents(docs)


class DocumentRetriever(BaseRetriever):
    """A retriever that contains the top k documents that contain the user query."""
    documents: List[Document] = []
    k: int = 5

    def model_post_init(self, ctx: Any)-> None:
        self.store_documents(self.documents)

    @staticmethod
    def store_documents(docs: List[Document]) -> None:
        """Add documents to the vector store."""
        splits = split_documents(docs)
        VECTOR_STORE.add_documents(splits)

    def add_uploaded_docs(self, uploaded_files):
        """Add uploaded documents. 接收 [(name, bytes), ...]"""
        docs = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, content in uploaded_files:
                temp_filepath = os.path.join(temp_dir, name)
                # Write file content first
                with open(temp_filepath, "wb") as f:
                    f.write(content)
                # Load document AFTER file is closed
                try:
                    docs.extend(load_document(temp_filepath))
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
                    continue

        self.documents.extend(docs)
        self.store_documents(docs)

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        if len(self.documents) == 0:
            return []
        return VECTOR_STORE.similarity_search(query=query, k=self.k)