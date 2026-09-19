from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import interrupt, Command
from typing import Optional

class State(MessagesState):
    home_address: Optional[str]

def _human_input(state: State):
    address = interrupt("What is your address?")
    return {"home_address": address}

builder = StateGraph(State)
builder.add_node("human_input", _human_input)
builder.add_edge(START, "human_input")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream({"messages": [("human", "What is weather today?")]}, config):
    print(chunk)

for chunk in graph.stream(Command(resume="Munich"), config):
    print(chunk)