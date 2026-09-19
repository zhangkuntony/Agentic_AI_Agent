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

i = 6
ds_dict = ds["test"].take(100).to_dict()
question = ds_dict["question"][i]
options = "\n".join([f"{i}. {a}" for i, a in enumerate(ds_dict["choices"][i])])
print(options)

# 2. Create research agent
from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
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

research_agent = create_agent(model=llm, tools=research_tools, system_prompt=system_prompt)

reflection_prompt = (
    "You are a university professor and you're supervising a student who is "
    "working on multiple-choice exam question. "
    "Given the dialogue above, reflect on the answer provided and give a feedback "
    " if needed. If you think the final answer is correct, reply with "
    "an empty message. Only provide critique if you think the last answer might "
    "be incorrect or there are reasoning flaws. Do not assume anything, "
    "evaluate only the reasoning the student provided and whether there is "
    "enough evidence for their answer."
)

# 3. Put everything together to create out agents
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

question_template = PromptTemplate.from_template(
    "QUESTION:\n{question}\n\nANSWER OPTIONS:\n{options}\n\n"
)

def _ask_question(state):
    return {"messages": [("human", question_template.invoke(state).text)]}

def _give_feedback(state, config: RunnableConfig):
    messages = state["messages"] + [("human", reflection_prompt)]
    max_messages = config["configurable"].get("max_messages", 20)

    if len(messages) > max_messages:
        return Command(
            update={},
            goto=END
        )

    result = llm.invoke(messages)

    if result.content:
        return Command(
            update={"messages": [
                ("assistant", result.content),
                ("human", "Please, address the feedback above and give an answer.")
            ]},
            goto="research"
        )

    return Command(
        update={},
        goto=END
    )

from langgraph.graph import StateGraph, START, END, MessagesState

class ReflectionAgentState(MessagesState):
    question: str
    options: str

builder = StateGraph(ReflectionAgentState)
builder.add_node("ask_question", _ask_question)
builder.add_node("research", research_agent)
builder.add_node("reflect", _give_feedback)

builder.add_edge(START, "ask_question")
builder.add_edge("ask_question", "research")
builder.add_edge("research", "reflect")
graph = builder.compile()

from utils.utils import show_graph

show_graph(graph)

# 4. Run agent and see output
import asyncio

async def main():
    async for _, event in graph.astream({"question": question, "options": options}, stream_mode=["values"]):
        print(len(event["messages"]))
        for m in event["messages"]:
            print(type(m))
            m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())
