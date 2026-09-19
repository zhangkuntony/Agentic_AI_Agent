from datasets import load_dataset, load_from_disk
from pathlib import Path


# 1. Load Dataset from HuggingFace
DATASET_NAME = "cais/mmlu"
DATASET_CONFIG = "high_school_geography"

# local file folder: src/chapter06/data/mmlu_high_school_geography
DATA_DIR = Path(__file__).resolve().parent / "data" / f"{DATASET_NAME.split('/')[-1]}_{DATASET_CONFIG}"

def load_mmlu():
    """Only download at the first time and save data into local folder. And will load from local folder later"""
    if DATA_DIR.exists():
        print(f"[local] Loading local dataset from {DATA_DIR}")
        return load_from_disk(str(DATA_DIR))

    print(f"[hugging face] No local dataset, will download from HuggingFace for the first time: {DATASET_NAME}/{DATASET_CONFIG}")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATA_DIR))
    print(f"[save] Saved to local: {DATA_DIR}")

    return ds

ds = load_mmlu()

ds_dict = ds["test"].take(100).to_dict()
print(ds_dict["question"][0])

print(ds_dict["choices"][0])

print(ds_dict["choices"][0][ds_dict["answer"][0]])

# 2. Create research agent
from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain.agents import AgentState, create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

research_tools = load_tools(
    tool_names=["ddg-search"],
    llm=llm,
)

system_prompt = (
    "You're a hard-working, curious and creative student. "
    "You're working on exam question. Think step by step."
    "Always provide an argumentation for your answer. "
    "Do not assume anything, use available tools to search "
    "for evidence and supporting statements."
)

class ResearchState(AgentState):
    question: str
    options: str

research_agent = create_agent(model=llm, tools=research_tools, state_schema=ResearchState, system_prompt=system_prompt)


# 3. Create another agent that reflects on the answer and provides critique

class ReflectionState(ResearchState):
    answer: str
    feedback: str

research_agent_with_critique = create_agent(model=llm, tools=research_tools, state_schema=ReflectionState, system_prompt=system_prompt)


from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, TypedDict

reflection_prompt = (
    "You are a university professor and you're supervising a student who is "
    "working on a multiple-choice exam question."
    "\nQUESTION: {question}.\nANSWER OPTIONS:\n{options}\n."
    "STUDENT'S ANSWER:\n{answer}\n"
    "Reflect on the student's reasoning and decide whether the answer is right "
    "or wrong. Respond with exactly one of the two fields, never both:\n"
    "- If the answer is correct and sufficiently supported by the reasoning, "
    "fill in 'answer' with the final answer and leave 'critique' empty.\n"
    "- If the answer might be incorrect or the reasoning has flaws, fill in "
    "'critique' with actionable feedback and leave 'answer' empty.\n"
    "Do not assume anything; evaluate only the reasoning the student provided "
    "and whether there is enough evidence for their answer."
)

class Response(BaseModel):
    """A final response to the user."""

    answer: Optional[str] = Field(
        description="The final answer. Provide it only when the student's answer is correct; leave it empty whenever a critique is given.",
        default=None,
    )
    critique: Optional[str] = Field(
        description="Actionable feedback on the initial answer. Provide it only when the answer might be incorrect or the reasoning is flawed; leave it empty whenever a final answer is given.",
        default=None,
    )


# 4. Put everything together and create a multi-agent system
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from operator import add

class ReflectionAgentState(TypedDict):
    question: str
    options: str
    answer: str
    steps: Annotated[int, add]
    response: Response


def _should_end(state: ReflectionAgentState, config: RunnableConfig) -> Literal["research", END]:
    max_reasoning_steps = config["configurable"].get("max_reasoning_steps", 10)
    # Only stop when we have a final answer and no outstanding critique.
    # If both fields are set, the answer still needs refinement, so keep researching.
    if state.get("response") and state["response"].answer and not state["response"].critique:
        return END
    if state.get("steps", 1) > max_reasoning_steps:
        return END
    return "research"


reflection_chain = PromptTemplate.from_template(reflection_prompt) | llm.with_structured_output(Response)


raw_prompt_template = (
    "Answer the following multiple-choice question. "
    "\nQUESTION:\n{question}\n\nANSWER OPTIONS:\n{options}\n"
)

raw_prompt_template_with_critique = (
    "You tried to answer the exam question and you get feedback from your "
    "professor. Work on improving your answer and incorporating the feedback. "
    "\nQUESTION:\n{question}\n\nANSWER OPTIONS:\n{options}\n\n"
    "INITIAL ANSWER:\n{answer}\n\nFEEDBACK:\n{feedback}"

)

def _reflection_step(state):
    result = reflection_chain.invoke(state)
    return {"response": result, "steps": 1}

def _research_start(state):
    user_content = raw_prompt_template.format(
        question=state["question"],
        options=state["options"],
    )
    answer = research_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    return {"answer": answer["messages"][-1].content}

def _research(state):
    user_content = raw_prompt_template_with_critique.format(
        question=state["question"],
        options=state["options"],
        answer=state["answer"],
        feedback=state["response"].critique,
    )
    answer = research_agent_with_critique.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    return {"answer": answer["messages"][-1].content}


builder = StateGraph(ReflectionAgentState)
builder.add_node("research_start", _research_start)
builder.add_node("research", _research)
builder.add_node("reflect", _reflection_step)

builder.add_edge(START, "research_start")
builder.add_edge("research_start", "reflect")
builder.add_edge("research", "reflect")
builder.add_conditional_edges("reflect", _should_end)
graph = builder.compile()

from utils.utils import show_graph

show_graph(graph)

# 5. Run agent and see output
import asyncio

i = 3
question = ds_dict["question"][i]
options = "\n".join([f"{i}. {a}" for i, a in enumerate(ds_dict["choices"][i])])

async def main():
    async for _, event in graph.astream({"question": question, "options": options}, stream_mode=["updates"]):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())
