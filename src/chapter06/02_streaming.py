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

# 2. Create a simple research agent
from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain.agents import AgentState, create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
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


# 3. Prepare a question and answer options

i = 6
ds_dict = ds["test"].take(100).to_dict()
question = ds_dict["question"][i]
options = "\n".join([f"{i}. {a}" for i, a in enumerate(ds_dict["choices"][i])])


# 4. Run agent and see output
import asyncio

formatted_question = (
    "Answer the following multiple-choice question. "
    f"\nQUESTION:\n{question}\n\nANSWER OPTIONS:\n{options}\n"
)

async def main():
    input_msg = {"messages": [HumanMessage(content=formatted_question)]}

    async for _, event in research_agent.astream(input_msg, stream_mode=["values"]):
        print(len(event["messages"]))

    async for _, event in research_agent.astream(input_msg, stream_mode=["updates"]):
        node = list(event.keys())[0]
        print(node, len(event[node].get("messages", [])))

    async for _, event in research_agent.astream(input_msg, stream_mode=["updates"]):
        print(event)

    seen_events = set([])
    async for event in research_agent.astream_events(input_msg, version="v1"):
        if event["event"] not in seen_events:
            seen_events.add(event["event"])

    print(seen_events)

if __name__ == "__main__":
    asyncio.run(main())
