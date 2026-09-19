# 1. Start from the planning agent

import operator

from collections import deque
from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Annotated, Optional, TypedDict

# planner_llm = ChatOpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL,
#     model=MODEL_NAME,
#     temperature=1.0
# )
#
# class Plan(BaseModel):
#     """Plan to follow in future"""
#
#     steps: list[str] = Field(
#         description="different steps to follow, should be in sorted order"
#     )
#
# system_prompt_template = (
#     "For the given task, come up with a step by step plan.\n"
#     "This plan should involve individual tasks, that if executed correctly will "
#     "yield the correct answer. Do not add any superfluous steps.\n"
#     "The result of the final step should be the final answer. Make sure that each "
#     "step has all the information needed - do not skip steps."
# )
#
# planner_prompt = ChatPromptTemplate.from_messages(
#     [("system", system_prompt_template),
#      ("user", "Prepare a plan how to solve the following task:\n{task}\n")]
# )
#
# planner = planner_prompt | planner_llm.with_structured_output(Plan)
#
#
# # 2. Build an agent that executes a step in our plan
#
# execution_llm = ChatOpenAI(
#     api_key=API_KEY,
#     base_url=BASE_URL,
#     model=MODEL_NAME,
# )
#
# tools = load_tools(
#     tool_names=["ddg-search", "arxiv", "wikipedia"],
#     llm=execution_llm
# )
#
# system_prompt = (
#     "You're a smart assistant that carefully helps to solve complex tasks.\n"
#     " Given a general plan to solve a task and a specific step, work on this step. "
#     " Don't assume anything, keep in minds things might change and always try to "
#     "use tools to double-check yourself.\nUse Search to gather "
#     "information about common facts, fresh events and news, use Arxiv to get "
#     "ideas on recent research and use Wikipedia for common knowledge."
# )
#
# step_template = (
#     "Given the task and the plan, try to execute on a specific step of the plan.\n"
#     "TASK:\n{task}\n\nPLAN:\n{previous_steps}\n\nSTEP TO EXECUTE:\n{step}\n"
# )
#
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("user", step_template),
# ])
#
# execution_agent = prompt_template | create_agent(model=execution_llm, tools=tools)
#
#
# # 3. Define a separate structure to keep track of the tree of options
#
# class TreeNode:
#
#     def __init__(
#             self,
#             node_id: int,
#             step: str,
#             step_output: Optional[str] = None,
#             parent: Optional["TreeNode"] = None,
#     ):
#         self.node_id = node_id
#         self.step = step
#         self.step_output = step_output
#         self.parent = parent
#         self.children = []
#         self.final_response = None
#
#     def __repr__(self):
#         parent_id = self.parent.node_id if self.parent else None
#         return f"Node_id: {self.node_id}, parent: {parent_id}, {len(self.children)} children"
#
#     def get_full_plan(self) -> str:
#         """Returns formatted plan with step numbers and past result"""
#         steps = []
#         node = self
#         while node.parent:
#             steps.append((node.step, node.step_output))
#             node = node.parent
#
#         full_plan = []
#         for i, (step, result) in enumerate(steps[::-1]):
#             if result:
#                 full_plan.append(f"# {i+1}. Planned step: {step}\nResult: {result}\n")
#         return "\n".join(full_plan)
#
#
# # 4. Define additional states
#
# class PlanEvaluation(BaseModel):
#
#     is_final: bool = Field(
#         description="Whether the task is solved or not. Plan is not final if additional steps are required.",
#         default=False
#     )
#
#     score: float = Field(
#         description="A score from 0. to 1.0 that evaluates the quality of the plan and how probable it is to solve the task with this plan.",
#         default=False
#     )
#
# class PlanState(TypedDict):
#     task: str
#     root: TreeNode
#     queue: deque[TreeNode]
#     current_node: TreeNode
#     next_node: TreeNode
#     is_current_node_final: bool
#     paths_explored: Annotated[int, operator.add]
#     visited_ids: set[int]
#     max_id: int
#     candidates: Annotated[list[str], operator.add]
#     best_candidate: str
#
#
# # 5. Create a replanning agent than takes into account the previous steps and adjusts the plan accordingly
#
# class ReplanStep(BaseModel):
#     """Replanned next step in the plan."""
#
#     steps: list[str] = Field(
#         description="different options of the proposed next step"
#     )
#
# llm_replanner = execution_llm.with_structured_output(ReplanStep)
#
# replanner_prompt_template = (
#     "Suggest next action in the plan. Do not add any superfluous steps.\n"
#     "If you think no actions are needed, just return an empty list of steps. "
#     "TASK: {task}\n PREVIOUS STEPS WITH OUTPUTS: {current_plan}"
# )
#
# replanner_prompt = ChatPromptTemplate.from_messages(
#     [("system", "You're a helpful assistant. Your goal is to help with planning actions to solve the task. Do not solve the task itself."),
#      ("user", replanner_prompt_template)]
# )
#
# replanner = replanner_prompt | llm_replanner


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

