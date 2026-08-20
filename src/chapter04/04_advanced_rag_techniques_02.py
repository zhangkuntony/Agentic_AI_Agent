from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utils.ark_embeddings import ArkMultimodalEmbeddings

# 1. Source Attribution

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0
)

# Example documents
documents = [
    Document(
        page_content="The transformer architecture was introduced in the paper 'Attention is All You Need' by Vaswani et al. in 2017.",
        metadata={"source": "Neural Network Review 2021", "page": 42}
    ),
    Document(
        page_content="BERT uses bidirectional training of the Transformer, masked language modeling, and next sentence prediction tasks.",
        metadata={"source": "Introduction to NLP", "page": 137}
    ),
    Document(
        page_content="GPT models are autoregressive transformers that predict the next token based on previous tokens.",
        metadata={"source": "Large Language Models Survey", "page": 89}
    )
]

# Create a vector store and retriever
embeddings = ArkMultimodalEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Source attribution prompt template
attribution_prompt = ChatPromptTemplate.from_template("""
You are a precise AI assistant that provides well-sourced information.
Answer the following question based ONLY on the provided sources. For each fact or claim in your answer,
include a citation using [1], [2], etc. that refers to the source. Include a numbered reference list at the end.

Question: {question}

Sources:
{sources}

Your answer:
""")

# Create a source-formatted string from documents
def format_sources_with_citations(docs):
    formatted_sources = []
    for i, doc in enumerate(docs, 1):
        source_info = f"[{i}] {doc.metadata.get('source', 'Unknown source')}"
        if doc.metadata.get('page'):
            source_info += f", page {doc.metadata['page']}"
        formatted_sources.append(f"{source_info}\n{doc.page_content}")
    return "\n\n".join(formatted_sources)

# Build the rag chain with source attribution
def generate_attribution_response(question):
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Format sources with citation numbers
    sources_formatted = format_sources_with_citations(retrieved_docs)

    # Create the attribution chain using LCEL
    attribution_chain = attribution_prompt | llm | StrOutputParser()

    # Generate the response with citations
    response = attribution_chain.invoke({
        "question": question,
        "sources": sources_formatted
    })

    return response

# Example usage
question = "How do transformer models work and what are some examples?"
attributed_answer = generate_attribution_response(question)
print(attributed_answer)
