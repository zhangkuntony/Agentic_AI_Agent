from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import START, StateGraph, add_messages
from typing import Annotated
from typing_extensions import List, TypedDict

from chapter04.chat_bot.llms import chat_model
from chapter04.chat_bot.retriever import DocumentRetriever


system_prompt = (
    "You're a helpful AI assistant. Given a user question "
    "and some corporate document snippets, write documentation. "
    "If none of the documents is relevant to the question, "
    "mention that there's no relevant document, and then "
    "answer the question to the best of your knowledge."
    "\n\nHere are the corporate documents: "
    "{context}"
)

retriever = DocumentRetriever()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    issues_report: str
    issues_detected: bool
    messages: Annotated[list, add_messages]


# Define application steps
def retrieve(state: State):
    retrieved_docs = retriever.invoke(state["messages"][-1].content)
    print(retrieved_docs)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke(
        {"question": state["messages"][-1].content, "context": docs_content}
    )
    response = chat_model.invoke(messages)
    print(response.content)
    return {"answer": response.content}


def double_check(state: State):
    result = chat_model.invoke([{
        "role": "user",
        "content": (
            f"Review the following project documentation for compliance with our corporate standards. "
            f"Return 'ISSUES FOUND' followed by any issues detected or 'NO ISSUES': {state['answer']}"
        )
    }])

    # Extract actual response (after thinking block)
    content = result.content
    if "</think>" in content:
        actual_response = content.split("</think>", 1)[1].strip()
    else:
        actual_response = content.strip()

    if "ISSUES FOUND" in actual_response:
        print("issues detected")
        return {
            "issues_report": actual_response.split("ISSUES FOUND", 1)[1].split(),
            "issues_detected": True
        }
    print("no issues detected")
    return {
        "issues_report": "",
        "issues_detected": False
    }


# NODE: doc_finalizer
# Finalizes the documentation by incorporating feedback if available.
def doc_finalizer(state: State):
    """Finalize documentation by integrating human feedback."""
    if "issues_detected" in state and state["issues_detected"]:
        response = chat_model.invoke([{
            "role": "user",
            "content": (
                f"Revise the following documentation to address these feedback points: {state['issues_report']}\n"
                f"Original Document: {state['answer']}\n"
                f"Always return the full revised document, even if no changes are needed."
            )
        }])
        return {
            "messages": [AIMessage(response.content)]
        }
    return {
        "messages": [AIMessage(state['answer'])]
    }


# Compile application and test
graph_builder = StateGraph(State).add_sequence(
    [retrieve, generate, double_check, doc_finalizer]
)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("doc_finalizer", END)
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}