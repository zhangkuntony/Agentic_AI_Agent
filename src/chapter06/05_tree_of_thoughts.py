# 1. Start from the planning agent

import operator

from collections import deque
from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, TypedDict

planner_llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=1.0
)

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
    "step has all the information needed - do not skip steps."
)

planner_prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt_template),
     ("user", "Prepare a plan how to solve the following task:\n{task}\n")]
)

planner = planner_prompt | planner_llm.with_structured_output(Plan)


# 2. Build an agent that executes a step in our plan

execution_llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

tools = load_tools(
    tool_names=["ddg-search"],
    llm=execution_llm
)

system_prompt = (
    "You're a smart assistant that carefully helps to solve complex tasks.\n"
    " Given a general plan to solve a task and a specific step, work on this step. "
    " Don't assume anything, keep in minds things might change and always try to "
    "use tools to double-check yourself.\nUse Search to gather "
    "information about common facts, fresh events and news, use Arxiv to get "
    "ideas on recent research and use Wikipedia for common knowledge."
)

step_template = (
    "Given the task and the plan, try to execute on a specific step of the plan.\n"
    "TASK:\n{task}\n\nPLAN:\n{previous_steps}\n\nSTEP TO EXECUTE:\n{step}\n"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", step_template),
])

execution_agent = prompt_template | create_agent(model=execution_llm, tools=tools)


# 3. Define a separate structure to keep track of the tree of options

class TreeNode:

    def __init__(
            self,
            node_id: int,
            step: str,
            step_output: Optional[str] = None,
            parent: Optional["TreeNode"] = None,
    ):
        self.node_id = node_id
        self.step = step
        self.step_output = step_output
        self.parent = parent
        self.children = []
        self.final_response = None

    def __repr__(self):
        parent_id = self.parent.node_id if self.parent else None
        return f"Node_id: {self.node_id}, parent: {parent_id}, {len(self.children)} children"

    def get_full_plan(self) -> str:
        """Returns formatted plan with step numbers and past result"""
        steps = []
        node = self
        while node.parent:
            steps.append((node.step, node.step_output))
            node = node.parent

        full_plan = []
        for i, (step, result) in enumerate(steps[::-1]):
            if result:
                full_plan.append(f"# {i+1}. Planned step: {step}\nResult: {result}\n")
        return "\n".join(full_plan)


# 4. Define additional states

class PlanEvaluation(BaseModel):

    is_final: bool = Field(
        description="Whether the task is solved or not. Plan is not final if additional steps are required.",
        default=False
    )

    score: float = Field(
        description="A score from 0. to 1.0 that evaluates the quality of the plan and how probable it is to solve the task with this plan.",
        default=0.0
    )

class PlanState(TypedDict):
    task: str
    root: TreeNode
    queue: deque[TreeNode]
    current_node: TreeNode
    next_node: TreeNode
    is_current_node_final: bool
    paths_explored: Annotated[int, operator.add]
    visited_ids: set[int]
    max_id: int
    candidates: Annotated[list[str], operator.add]
    best_candidate: str


# 5. Create a replanning agent than takes into account the previous steps and adjusts the plan accordingly

class ReplanStep(BaseModel):
    """Replanned next step in the plan."""

    steps: list[str] = Field(
        description="different options of the proposed next step"
    )

llm_replanner = execution_llm.with_structured_output(ReplanStep)

replanner_prompt_template = (
    "Suggest next action in the plan. Do not add any superfluous steps.\n"
    "If you think no actions are needed, just return an empty list of steps. "
    "TASK: {task}\n PREVIOUS STEPS WITH OUTPUTS: {current_plan}"
)

replanner_prompt = ChatPromptTemplate.from_messages(
    [("system", "You're a helpful assistant. Your goal is to help with planning actions to solve the task. Do not solve the task itself."),
     ("user", replanner_prompt_template)]
)

replanner = replanner_prompt | llm_replanner


# 6. Need a final step that includes a major voting for the best candidate

prompt_voting = PromptTemplate.from_template(
    "Pick the best solution for a given task. "
    "\nTASK:{task}\n\nSOLUTIONS:\n{candidates}\n"
)

def _vote_for_the_best_option(state):
    candidates = state.get("candidates", [])
    if not candidates:
        return {"best_response": None}
    all_candidates = [f"OPTION {i+1}: {c}" for i, c in enumerate(candidates)]

    response_schema = {
        "title": "VotingResult",
        "description": "Pick the best option for the given task",
        "type": "object",
        "properties": {
            "best_option": {
                "type": "string",
                "enum": [str(i + 1) for i in range(len(all_candidates))],
            }
        },
        "required": ["best_option"],
    }

    llm_enum = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL_NAME
    ).with_structured_output(response_schema)

    result = (prompt_voting | llm_enum).invoke(
        {"candidates": "\n".join(all_candidates), "task": state["task"]}
    )
    return {"best_candidate": candidates[int(result["best_option"])-1]}

test_result = _vote_for_the_best_option({"candidates": ["1", "5", "4"], "task": "How much is 2+2?"})
print(test_result)


# 7. Put everything together and create agent

final_prompt = PromptTemplate.from_template(
    "You're a helpful assistant that has executed on a plan."
    "Given the results of the execution, prepare the final response.\n"
    "Don't assume anything\nTASK:\n{task}\n\nPLAN WITH RESUlTS:\n{plan}\n"
    "FINAL RESPONSE:\n"
)

responder = final_prompt | execution_llm | StrOutputParser()

async def _build_initial_plan(state: PlanState) -> PlanState:
    plan = await planner.ainvoke(state["task"])
    queue = deque()
    root = TreeNode(step=plan.steps[0], node_id=1)
    queue.append(root)
    current_root = root
    max_id = len(plan.steps)
    for idx, step in enumerate(plan.steps[1:], start=2):
        child = TreeNode(node_id=idx, step=step, parent=current_root)
        current_root.children.append(child)
        queue.append(child)
        current_root = child
    return {"root": root, "queue": queue, "max_id": max_id}

async def _run_node(state: PlanState, config: RunnableConfig):
    node = state.get("next_node")
    visited_ids = state.get("visited_ids", set())
    queue = state["queue"]
    if node is None:
        while queue and not node:
            node = state["queue"].popleft()
            if node.node_id in visited_ids:
                node = None
        if not node:
            return Command(goto="vote", update={})

    print(f"[run] Executing node {node.node_id}: {node.step[:60]}...", flush=True)
    step = await execution_agent.ainvoke({
        "previous_steps": node.get_full_plan(),
        "step": node.step,
        "task": state["task"]
    })

    node.step_output = step["messages"][-1].content
    visited_ids.add(node.node_id)
    return {
        "current_node": node,
        "queue": queue,
        "visited_ids": visited_ids,
        "next_node": None,
        "paths_explored": 1,
    }

async def _plan_next(state: PlanState, config: RunnableConfig) -> PlanState:
    max_candidates = config["configurable"].get("max_candidates", 1)
    node = state["current_node"]
    next_step = await replanner.ainvoke({"task": state["task"], "current_plan": node.get_full_plan()})
    if not next_step.steps:
        return {"is_current_node_final": True}
    max_id = state["max_id"]
    for step in next_step.steps[:max_candidates]:
        child = TreeNode(node_id=max_id+1, step=step, parent=node)
        max_id += 1
        node.children.append(child)
        state["queue"].append(child)
    return {"is_current_node_final": False, "next_node": child, "max_id": max_id}

async def _get_final_response(state: PlanState) -> PlanState:
    node = state["current_node"]
    final_response = await responder.ainvoke({"task": state["task"], "plan": node.get_full_plan()})
    node.final_response = final_response
    return {"candidates": [final_response]}

def _route_after_planning(state: PlanState, config: RunnableConfig) -> Literal["run", "generate_response", "vote"]:
    max_paths = config["configurable"].get("max_paths", 10)
    if state.get("paths_explored", 0) >= max_paths:
        print(f"[budget] Reached max_paths={max_paths}, stop exploring, will go to vote", flush=True)
        return "vote"
    return "generate_response" if state.get("is_current_node_final") else "run"

def _should_continue(state: PlanState, config: RunnableConfig) -> Literal["run", "vote"]:
    max_paths = config["configurable"].get("max_paths", 30)
    if state.get("paths_explored", 0) >= max_paths:
        return "vote"
    if state["queue"] or state.get("next_node"):
        return "run"
    return "vote"

builder = StateGraph(PlanState)
builder.add_node("initial_plan", _build_initial_plan)
builder.add_node("run", _run_node)
builder.add_node("plan_next", _plan_next)
builder.add_node("generate_response", _get_final_response)
builder.add_node("vote", _vote_for_the_best_option)

builder.add_edge(START, "initial_plan")
builder.add_edge("initial_plan", "run")
builder.add_edge("run", "plan_next")
builder.add_conditional_edges("plan_next", _route_after_planning)
builder.add_conditional_edges("generate_response", _should_continue)
builder.add_edge("vote", END)

graph = builder.compile()

from utils.utils import show_graph
show_graph(graph)

import asyncio

task = "Write a strategic one-pager of building an AI startup"

async def main():
    config = {
        "recursion_limit": 60,
        "configurable": {"max_paths": 10, "max_candidates": 1}
    }
    result = None

    async for event in graph.astream(
            {"task": task}, config=config, stream_mode="values"
    ):
        result = event
        queue = event.get("queue") or []
        print(
            f"[state] queue={len(queue):>2} "
            f"paths={event.get('paths_explored')} "
            f"final={event.get('is_current_node_final')} "
            f"candidates={len(event.get('candidates') or [])}",
            flush=True,
        )
    print("candidates: ", len(result["candidates"]))
    print(result["best_candidate"])

if __name__ == "__main__":
    asyncio.run(main())