import operator

from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from utils.utils import show_graph

model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

# ============================================================
# 1. 长文本素材：Lilian Weng "LLM Powered Autonomous Agents"
# ============================================================
LONG_TEXT = r"""
Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.

In a LLM-powered autonomous agent system, LLM functions as the agent's brain, complemented by several key components: Planning, Memory, and Tool use.

Planning involves subgoal and decomposition — the agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks. It also includes reflection and refinement — the agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps, thereby improving the quality of final results.

Chain of thought (CoT) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to "think step by step" to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS or DFS with each state evaluated by a classifier or majority vote.

Another quite distinct approach, LLM+P, involves relying on an external classical planner to do long-horizon planning. This approach utilizes the Planning Domain Definition Language (PDDL) as an intermediate interface. In this process, LLM translates the problem into "Problem PDDL", then requests a classical planner to generate a PDDL plan, and finally translates the PDDL plan back into natural language.

Self-reflection is a vital aspect that allows autonomous agents to improve iteratively by refining past action decisions and correcting previous mistakes. ReAct integrates reasoning and acting within LLM by extending the action space to be a combination of task-specific discrete actions and the language space. The ReAct prompt template incorporates explicit steps for LLM to think, roughly formatted as Thought → Action → Observation, repeated many times.

Reflexion is a framework to equip agents with dynamic memory and self-reflection capabilities to improve reasoning skills. Reflexion has a standard RL setup where the reward model provides a simple binary reward. After each action, the agent computes a heuristic and optionally may decide to reset the environment to start a new trial. Chain of Hindsight (CoH) encourages the model to improve on its own outputs by explicitly presenting it with a sequence of past outputs, each annotated with feedback.

Memory can be defined as the processes used to acquire, store, retain, and later retrieve information. There are several types of memory: Sensory Memory (the earliest stage, lasting up to a few seconds), Short-Term Memory (working memory, capacity of about 7 items, lasts 20-30 seconds), and Long-Term Memory (stores information from days to decades with essentially unlimited capacity).

In the context of LLM agents, we can roughly map: Sensory memory as learning embedding representations for raw inputs; Short-term memory as in-context learning, restricted by the finite context window length of Transformer; Long-term memory as the external vector store that the agent can attend to at query time, accessible via fast retrieval.

The external memory can alleviate the restriction of finite attention span. A standard practice is to save the embedding representation of information into a vector store database that can support fast maximum inner-product search (MIPS). To optimize retrieval speed, the common choice is the approximate nearest neighbors (ANN) algorithm to return approximately top k nearest neighbors.

Common ANN algorithms include: LSH (Locality-Sensitive Hashing) — introduces a hashing function such that similar input items are mapped to the same buckets with high probability; ANNOY (Approximate Nearest Neighbors Oh Yeah) — uses random projection trees; HNSW (Hierarchical Navigable Small World) — inspired by small world networks, builds hierarchical layers of small-world graphs; FAISS (Facebook AI Similarity Search) — applies vector quantization by partitioning the vector space into clusters; ScaNN (Scalable Nearest Neighbors) — uses anisotropic vector quantization.

Tool use is a remarkable and distinguishing characteristic of human beings. Equipping LLMs with external tools can significantly extend the model capabilities. MRKL (Modular Reasoning, Knowledge and Language) is a neuro-symbolic architecture for autonomous agents. A MRKL system contains a collection of "expert" modules and the general-purpose LLM works as a router to route inquiries to the best suitable expert module.

Both TALM (Tool Augmented Language Models) and Toolformer fine-tune a LM to learn to use external tool APIs. The dataset is expanded based on whether a newly added API call annotation can improve the quality of model outputs. ChatGPT Plugins and OpenAI API function calling are good examples of LLMs augmented with tool use capability working in practice.

HuggingGPT is a framework to use ChatGPT as the task planner to select models available in HuggingFace platform according to model descriptions and summarize the response based on execution results. The system comprises 4 stages: Task planning, Model selection, Task execution, and Response generation.

API-Bank is a benchmark for evaluating the performance of tool-augmented LLMs. It contains 53 commonly used API tools, a complete tool-augmented LLM workflow, and 264 annotated dialogues involving 568 API calls. This benchmark evaluates the agent's tool use capabilities at three levels: calling the API, retrieving the API, and planning beyond retrieve and call.

In case studies, ChemCrow is a domain-specific example where LLM is augmented with 13 expert-designed tools to accomplish tasks across organic synthesis, drug discovery, and materials design. Human evaluations with experts showed that ChemCrow outperforms GPT-4 by a large margin, indicating a potential problem with using LLM to evaluate its own performance on domains requiring deep expertise.

Generative Agents is a fun experiment where 25 virtual characters, each controlled by a LLM-powered agent, live and interact in a sandbox environment inspired by The Sims. The design combines LLM with memory, planning and reflection mechanisms. Memory stream records agents' experience; retrieval model surfaces context according to relevance, recency and importance; reflection mechanism synthesizes memories into higher level inferences over time.

This simulation results in emergent social behavior, such as information diffusion, relationship memory and coordination of social events like hosting parties and inviting others.

AutoGPT has drawn attention to setting up autonomous agents with LLM as the main controller. It has reliability issues given the natural language interface, but is a cool proof-of-concept demo. GPT-Engineer creates a whole repository of code given a task specified in natural language, instructed to think over smaller components to build and ask for user input to clarify questions.

Key challenges include: Finite context length — the restricted context capacity limits inclusion of historical information, detailed instructions, API call context, and responses. Challenges in long-term planning and task decomposition — LLMs struggle to adjust plans when faced with unexpected errors, making them less robust compared to humans. Reliability of natural language interface — current agent systems rely on natural language as interface between LLMs and external components, but model outputs may have formatting errors and occasionally exhibit rebellious behavior.
"""

class AgentState(TypedDict):
    text: str
    chunk_size: int
    overlap: int
    summaries: Annotated[list, operator.add]
    final_summary: str

class _ChunkState(TypedDict):
    chunk_text: str

# ============================================================
# Map prompt
# ============================================================
map_prompt = ChatPromptTemplate.from_messages([
    ("human", """You are a precise text summarization assistant.
        Read the following text segment from a technical article about LLM-powered AI agents.
        Write a concise summary (2-3 sentences) capturing ONLY the key points from this segment.
        Do NOT add any information not present in the text.

        Text segment:
        {text}

        Summary:""")
])

# ============================================================
# Reduce prompt
# ============================================================
reduce_prompt = PromptTemplate.from_template(
    "You are given a list of summaries from different sections of an article about LLM-powered AI agents.\n"
    "SUMMARIES:\n{summaries}\n"
    "Based on that, prepare a comprehensive summary of the whole text."
)

async def _summarize_text_chunk(state: _ChunkState):
    """Map 节点：对单个文本片段做摘要"""
    chain = map_prompt | model | StrOutputParser()
    summary = await chain.ainvoke({"text": state["chunk_text"]})
    return {"summaries": [summary]}

def _map_summaries(state: AgentState):
    """条件边函数：切分文本，为每个片段生成 Send，触发并行 Map"""
    chunks = []
    start = 0
    while start < len(state["text"]):
        end = min(start + state["chunk_size"], len(state["text"]))
        chunks.append(state["text"][start:end])
        start += state["chunk_size"] - state["overlap"]

    return [
        Send("summarize_text_chunk", {"chunk_text": chunk}) for chunk in chunks
    ]

# ============================================================
# 辅助函数：合并摘要
# ============================================================
def _merge_summaries(summaries: list[str]) -> str:
    parts = []
    for i, s in enumerate(summaries, 1):
        parts.append(f"Summary {i}: \n{s}")
    return "\n\n".join(parts)

async def _generate_final_summary(state: AgentState):
    """Reduce 节点：汇总所有局部摘要，生成最终摘要"""
    merged = _merge_summaries(state["summaries"])
    chain = reduce_prompt | model | StrOutputParser()
    final = await chain.ainvoke({"summaries": merged})
    return {"final_summary": final}

# ============================================================
# 构建图
# ============================================================
graph = StateGraph(AgentState)

graph.add_node("summarize_text_chunk", _summarize_text_chunk)
graph.add_node("generate_final_summary", _generate_final_summary)

graph.add_conditional_edges(START, _map_summaries, ["summarize_text_chunk"])
graph.add_edge("summarize_text_chunk", "generate_final_summary")
graph.add_edge("generate_final_summary", END)

app = graph.compile()
show_graph(app)

# ============================================================
# 运行
# ============================================================
import asyncio

async def main():
    result = await app.ainvoke({
        "text": LONG_TEXT,
        "chunk_size": 1500,
        "overlap": 200,
        "summaries": [],
        "final_summary": "",
    })

    print(f"Generated {len(result['summaries'])} summaries")
    for i, s in enumerate(result['summaries'], 1):
        print(f"--- Summary {i} ---")
        print(f"{s}\n")

    print(f"final summary {result['final_summary']}")

asyncio.run(main())
