from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from utils.utils import show_graph

# 1. 简单的状态管理示例
class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str

def analyze_job_description(state):
    print("...Analyzing a provided job description ...")
    return {"is_suitable": len(state["job_description"]) > 100}

def generate_application(state):
    print("...generating application ...")
    return {"application": "some_fake_application"}

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)

builder.add_edge(START, "analyze_job_description")
builder.add_edge("analyze_job_description", "generate_application")
builder.add_edge("generate_application", END)

graph = builder.compile()

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

from langchain_core.runnables import Runnable
print(isinstance(graph, Runnable))

show_graph(graph)

res = graph.invoke({"job_description": "fake_id"})
print(res)

# 2. 带有条件边的状态管理

from typing import Literal

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)

def is_suitable_condition(state: JobApplicationState) -> Literal["generate_application", END]:
    if state.get("is_suitable"):
        return "generate_application"
    return END

builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

show_graph(graph)