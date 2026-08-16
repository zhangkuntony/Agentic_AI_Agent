from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import trim_messages, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

class PrintOutputCallback(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print(f"Amount of input messages: {len(messages)}")

sessions = {}
handler = PrintOutputCallback()
llm = FakeListChatModel(responses=["ai1", "ai2", "ai3"])

def get_session_history(session_id: str):
    if session_id not in sessions:
        sessions[session_id] = InMemoryChatMessageHistory()
    return sessions[session_id]

trimmer = trim_messages(
    max_tokens=1,
    strategy="last",
    token_counter=len,
    include_system=True,
    start_on="human",
)

raw_chain = trimmer | llm
chain = RunnableWithMessageHistory(raw_chain, get_session_history)

config = {"callbacks": [PrintOutputCallback()], "configurable": {"session_id": "1"}}
_ = chain.invoke(
    [HumanMessage("Hi!")],
    config=config,
)
print(f"History length: {len(sessions['1'].messages)}")

_ = chain.invoke(
    [HumanMessage("How are you?")],
    config=config,
)
print(f"History length: {len(sessions['1'].messages)}")

print(sessions['1'].messages)

print(trimmer.invoke(sessions['1'].messages))


from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, MessagesState

def test_node(state: MessagesState):
    # ignore the last message since it's an input one
    messages = state["messages"]
    print(f"History length= {len(messages[:-1])}")
    return {"messages": [AIMessage(content="Hello!")]}

builder = StateGraph(MessagesState)
builder.add_node("test_node", test_node)
builder.add_edge(START, "test_node")
builder.add_edge("test_node", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

_ = graph.invoke({"messages": [HumanMessage(content="test")]}, config={"configurable": {"thread_id": "thread-a"}})
_ = graph.invoke({"messages": [HumanMessage(content="test")]}, config={"configurable": {"thread_id": "thread-b"}})
_ = graph.invoke({"messages": [HumanMessage(content="test")]}, config={"configurable": {"thread_id": "thread-a"}})

checkpoints = list(memory.list(config={"configurable": {"thread_id": "thread-a"}}))
for check_point in checkpoints:
    print(check_point.config["configurable"]["checkpoint_id"])

checkpoint_id = checkpoints[-1].config["configurable"]["checkpoint_id"]
_ = graph.invoke({"messages": [HumanMessage(content="test")]}, config={"configurable": {"thread_id": "thread-a", "checkpoint_id": checkpoint_id}})

checkpoint_id = checkpoints[-3].config["configurable"]["checkpoint_id"]
_ = graph.invoke({"messages": [HumanMessage(content="test")]}, config={"configurable": {"thread_id": "thread-a", "checkpoint_id": checkpoint_id}})
