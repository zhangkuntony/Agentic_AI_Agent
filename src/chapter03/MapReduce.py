import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

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
# 2. 长文本切片，并生成HumanMessages
# ============================================================
def _create_input_dicts(text: str, chunk_size: int = 1500, overlap: int = 200):
    """按字符数切分文本，返回包含 {text: chunk} 的 dict 列表"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [{"text": chunk} for chunk in chunks]

# input_dicts = _create_input_dicts(LONG_TEXT)
map_chain = map_prompt | model | StrOutputParser()

# summaries = map_chain.batch(input_dicts, config={"max_concurrency": 3})
# print(f"total {len(summaries)} summaries")
# for idx, summary in enumerate(summaries, 1):
#     print(f"summary {idx}: {summary}")

# 把切分步骤包装成 Runable，为后续使用LangGraph铺垫
create_inputs_chain = RunnableLambda(lambda x: _create_input_dicts(**x))
# 把批量Map步骤包装成 Runable，并串联
map_step_chain = create_inputs_chain | RunnableLambda(lambda x: map_chain.batch(x, config={"max_concurrency": 3}))
# 调用
summaries =  map_step_chain.invoke({"text": LONG_TEXT})
print(f"total {len(summaries)} summaries")
for idx, summary in enumerate(summaries, 1):
    print(f"summary {idx}: {summary}")


def _merge_summaries(summaries: list[str], **kwargs) -> str:
    sub_summaries = []
    for i, summary in enumerate(summaries, 1):
        sub_summary = (
            f"Summary {i}: \n{summary}\n"
        )
        sub_summaries.append(sub_summary)
    return "".join(sub_summaries)

reduce_prompt = PromptTemplate.from_template(
    "You are given a list of summaries.\n"
    "SUMMARIES: \n{summaries}\n"
    "Based on that, prepare a summary of a whole text."
)

reduce_chain = RunnableLambda(lambda x: _merge_summaries(**x)) | reduce_prompt | model | StrOutputParser()
# final_summary = reduce_chain.invoke({"summaries": summaries})
# print(f"final summary {final_summary}")

final_chain = (
    RunnablePassthrough.assign(summaries=map_step_chain).assign(final_summary=reduce_chain)
    | RunnableLambda(lambda x: x["final_summary"])
)

result = final_chain.invoke({"text": LONG_TEXT})
print(f"final result {result}")