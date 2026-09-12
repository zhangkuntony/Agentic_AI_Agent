import arxiv as arxiv_pkg
import asyncio
import math
import numexpr as ne
import operator
import socket

from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.tools import tool, convert_runnable_to_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Annotated, Literal, TypedDict
from utils.utils import show_graph

# 给所有底层网络请求兜底超时（wikipedia / arxiv / ddgs 内部都没有显式 timeout）
socket.setdefaulttimeout(20)

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: list[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

system_prompt_template = (
    "For the given task, come up with a step by step plan.\n"
    "This plan should involve individual tasks, that if executed correctly will "
    "yield the correct answer. Do not add any superfluous steps.\n"
    "The result of the final step should be the final answer. Make sure that each "
    "step has all the information needed - do not skip steps.\n"
    "Use at most 4 steps and keep each step concise so the plan is quick to execute."
)

planner_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt_template),
     ("user", "Prepare a plan how to solve the following task:\n{task}\n")]
)

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=1.0,
    timeout=120,  # 单次模型调用最多等 120s，避免无限期等待
)

planner = planner_prompt | llm.with_structured_output(Plan)

class CalculatorArgs(BaseModel):
    expression: str = Field(description="Mathematical expression to be evaluated")

def calculator(state: CalculatorArgs, config: RunnableConfig) -> str:
    expression = state["expression"]
    # 兜底常量，避免 configurable 里没传 math_constants 时 pi/e/i 未定义
    math_constants = config.get("configurable", {}).get("math_constants") or {
        "pi": math.pi,
        "e": math.e,
        "i": 1j,
    }
    result = ne.evaluate(expression.strip(), local_dict=math_constants)
    return str(result)

@tool
def arxiv_search(query: str) -> str:
    """Search arXiv for recent research papers. Use for academic/technical literature.

    Args:
        query: plaintext search query, e.g. "large language model agents"
    """
    try:
        # num_retries/delay_seconds 默认 3 次 + 每次 3s，网络不通时会白等很久
        client = arxiv_pkg.Client(num_retries=1, delay_seconds=0.5)
        search = arxiv_pkg.Search(query=query, max_results=3)
        docs = [
            f"Published: {r.updated.date()}\n"
            f"Title: {r.title}\n"
            f"Authors: {', '.join(a.name for a in r.authors)}\n"
            f"Summary: {r.summary}"
            for r in client.results(search)
        ]
    except Exception as e:
        return f"Arxiv search failed: {e}"
    return "\n\n".join(docs) if docs else "No good Arxiv Result was found"

calculator_with_retry = RunnableLambda(calculator).with_retry(
    wait_exponential_jitter=True,
    stop_after_attempt=2,  # 参数错误是确定性失败，没必要重试 3 次
)

calculator_tool = convert_runnable_to_tool(
    calculator_with_retry,
    name="calculator",
    description=(
        """
        Calculates a single mathematical expression, incl. complex numbers.
        
        Rules (Python syntax):
          - Use ** for exponentiation, NEVER ^ (e.g. (2+3*i)**2)
          - Always add * between a number and a constant/variable, examples:
              73i -> 73*i
              7pi**2 -> 7*pi**2
          - Available constants: pi, e, i (imaginary unit = 1j)
        """
    ),
    args_schema=CalculatorArgs,
    arg_types={"expression": str},
)

tools = load_tools(
    tool_names=["ddg-search", "wikipedia"],
    llm=llm,
) + [arxiv_search]

system_prompt = (
    "You're a smart assistant that carefully helps to solve complex tasks.\n"
    " Given a general plan to solve a task and a specific step, work on this step. "
    " Don't assume anything, keep in minds things might change and always try to "
    "use tools to double-check yourself.\n"
    " Use a calculator for mathematical computations, use Search to gather"
    "for information about common facts, fresh events and news, use Arxiv to get "
    "ideas on recent research and use Wikipedia for common knowledge."
)

step_template = (
    "Given the task and the plan, try to execute on a specific step of the plan.\n"
    "TASK:\n{task}\n\nPLAN:\n{plan}\n\nSTEP TO EXECUTE:\n{step}\n"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", step_template),
])

class StepState(AgentState):
    plan: str
    step: str
    task: str

def on_error(exc: Exception, request) -> str | None:
    return (
        f"`{request.tool_call['name']}` failed with {type(exc).__name__}. "
        "Try another tool or proceed without it."
    )

execution_agent = create_agent(
    model=llm,
    tools=tools + [calculator_tool],
    state_schema=StepState,
    system_prompt=system_prompt,
    middleware=[
        # 单个步骤内最多 6 次模型调用 / 8 次工具调用，防止在不可用的工具上无限重试
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=8, exit_behavior="continue"),
        # 工具异常转成错误消息交回模型，不再让整张图崩掉
        ToolErrorMiddleware(on_error),
    ],
)

class PlanState(TypedDict):
    task: str
    plan: Plan
    past_steps: Annotated[list[str], operator.add]
    final_response: str

def get_current_step(state: PlanState) -> int:
    """Returns the number of current step to be executed."""
    return len(state.get("past_steps", []))

def get_full_plan(state: PlanState) -> str:
    """Returns formatted plan with step numbers and past results."""
    full_plan = []
    for i, step in enumerate(state["plan"].steps):
        full_step = f"# {i+1}. Planned step: {step}\n"
        if i < get_current_step(state):
            full_step += f"Result: {state['past_steps'][i]}\n"
        full_plan.append(full_step)
    return "\n".join(full_plan)

final_prompt = PromptTemplate.from_template(
    "You're a helpful assistant that has executed on a plan."
    "Given the results of the execution, prepare the final response.\n"
    "Don't assume anything\nTASK:\n{task}\n\nPLAN WITH RESUlTS:\n{plan}\n"
    "FINAL RESPONSE:\n"
)

async def _build_initial_plan(state: PlanState) -> PlanState:
    plan = await planner.ainvoke(state["task"])
    return {"plan": plan}

async def _run_step(state: PlanState) -> PlanState:
    plan = state["plan"]
    current_step = get_current_step(state)
    print(f"\n[run] 执行第 {current_step + 1}/{len(plan.steps)} 步: {plan.steps[current_step]}")
    user_msg = step_template.format(
        task=state["task"],
        plan=get_full_plan(state),
        step=plan.steps[current_step],
    )
    result = await execution_agent.ainvoke({"messages": [("user", user_msg)]})
    return {"past_steps": [result["messages"][-1].content]}

async def _get_final_response(state: PlanState) -> PlanState:
    final_response = await (final_prompt | llm).ainvoke({"task": state["task"], "plan": get_full_plan(state)})
    return {"final_response": final_response}

def _should_continue(state: PlanState) -> Literal["run", "response"]:
    if get_current_step(state) < len(state["plan"].steps):
        return "run"
    return "response"

builder = StateGraph(PlanState)
builder.add_node("initial_plan", _build_initial_plan)
builder.add_node("run", _run_step)
builder.add_node("response", _get_final_response)

builder.add_edge(START, "initial_plan")
builder.add_edge("initial_plan", "run")
builder.add_conditional_edges("run", _should_continue)
builder.add_edge("response", END)

graph = builder.compile()

show_graph(graph)

async def main():
    task = "Write a strategic one-pager of building an AI startup?"

    # 只跑一遍：用 updates 模式边执行边打印，避免 ainvoke + astream 把整条流水线跑两次
    final_response = None
    async for output in graph.astream({"task": task}, stream_mode="updates"):
        for key, value in output.items():
            if key == "response":
                final_response = value["final_response"].content
            print(f"Output from node '{key}':")
            print("---")
            print(str(value)[:800])
        print("\n---\n")

    print("\n===== FINAL RESPONSE =====\n")
    print(final_response)

if __name__ == "__main__":
    asyncio.run(main())