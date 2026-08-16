from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from operator import add
from typing import Annotated, Optional, Union
from typing_extensions import TypedDict, Literal
from utils.utils import show_graph

# 1. StateGraph默认后写覆盖先写
class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str
    actions: list[str]

def analyze_job_description(state):
    print("...Analyzing a provided job description ...")
    result = {
        "is_suitable": len(state["job_description"]) < 100,
        "actions": ["action1"]
    }
    return result

def generate_application(state):
    print("...generating application ...")
    return {"application": "some_fake_application", "actions": ["action2"]}

def is_suitable_condition(state: JobApplicationState) -> Literal["generate_application", END]:
    if state.get("is_suitable"):
        return "generate_application"
    return END

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

show_graph(graph)

import asyncio

async def main():
    async for chunk in graph.astream(
        input={"job_description": "fake_jd"},
        stream_mode="values"
    ):
        print(chunk)
        print("\n\n")

# 观察这里的actions的值，可以发现后写入的action2覆盖了先写入的action1。因为还没有使用到reducer函数
asyncio.run(main())

# 2. 使用add方法作为reducer

class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str
    actions: Annotated[list[str], add]

def analyze_job_description(state):
    print("...Analyzing a provided job description ...")
    result = {
        "is_suitable": len(state["job_description"]) < 100,
        "actions": ["action1"]
    }
    return result

def generate_application(state):
    print("...generating application ...")
    return {"application": "some_fake_application", "actions": ["action2"]}

def is_suitable_condition(state: JobApplicationState) -> Literal["generate_application", END]:
    if state.get("is_suitable"):
        return "generate_application"
    return END

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

show_graph(graph)

import asyncio

async def main():
    async for chunk in graph.astream(
        input={"job_description": "fake_jd"},
        stream_mode="values"
    ):
        print(chunk)
        print("\n\n")

# 观察这里的actions的值，可以发现后写入的action2被添加到actions list中，跟在action1之后
asyncio.run(main())


# 3. 自定义my_reducer方法作为reducer

def my_reducer(left: list[str], right: Optional[Union[str, list[str]]]) -> list[str]:
    if right:
        return left + [right] if isinstance(right, str) else left + right
    return left


class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str
    actions: Annotated[list[str], my_reducer]

def analyze_job_description(state):
    print("...Analyzing a provided job description ...")
    result = {
        "is_suitable": len(state["job_description"]) < 100,
        "actions": "action1"
    }
    return result

def generate_application(state):
    print("...generating application ...")
    return {"application": "some_fake_application", "actions": ["action2", "action3"]}

def is_suitable_condition(state: JobApplicationState) -> Literal["generate_application", END]:
    if state.get("is_suitable"):
        return "generate_application"
    return END

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

show_graph(graph)

import asyncio

async def main():
    async for chunk in graph.astream(
        input={"job_description": "fake_jd"},
        stream_mode="values"
    ):
        print(chunk)
        print("\n\n")

# 观察这里的actions的值，可以发现后写入的actions 列表 ["action2, "action3"]被合并到actions list中，跟在action1之后
# 最终的actions的值为: ["action1", "action2", "action3"]
asyncio.run(main())


# 4. 使图可配置

def generate_application(state: JobApplicationState, config: RunnableConfig):
    model_provider = config["configurable"].get("model_provider", "Google")
    model_name = config["configurable"].get("model_name", "gemini-2.0-flash")
    print(f"...generating application with {model_provider} and {model_name} ...")
    return {"application": "some_fake_application", "actions": ["action2", "action3"]}

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))

show_graph(graph)

res = graph.invoke({"job_description": "fake_id"})
print(res)

res = graph.invoke({"job_description": "fake_id"}, config={"configurable": {"model_provider": "OpenAI", "model_name": "gpt-4o"}})
print(res)