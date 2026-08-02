import sys
from pathlib import Path

from utils import show_graph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import API_KEY, BASE_URL, MODEL_NAME
from enum import Enum
from langchain_classic.output_parsers import EnumOutputParser
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from operator import add
from typing import Annotated, Literal
from typing_extensions import TypedDict

import logging
logger = logging.getLogger(__name__)

model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

class IsSuitableJobEnum(Enum):
    YES = "YES"
    NO = "NO"

parser = EnumOutputParser(enum=IsSuitableJobEnum)

prompt_template_enum = (
    "Given a job description, decide whether it suites a junior Java developer."
    "\nJOB DESCRIPTION:\n{job_description}\n\nAnswer only YES or NO."
)

job_description = "test job description"

analyze_chain = model | parser


# 1. 添加一个try...except...代码块，来处理异常。当程序执行analyze_chain出现异常时，默认返回{"is_suitable": False}，不影响后续链路
def analyze_job_description(state):
    try:
      prompt = prompt_template_enum.format(job_description=job_description)
      result = analyze_chain.invoke(prompt)
      return {"is_suitable": result}
    except Exception as e:
      logger.error(f"Exception {e} occured while executing analyze_job_description")
      return {"is_suitable": False}

# 2. 创建一个MessagesIterator迭代器，每次执行奇数次时，抛出异常；执行偶数次时，正常返回AI消息
class MessagesIterator:

    def __init__(self):
        self._count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._count += 1
        if self._count % 2 == 1:
            raise ValueError("Something went wrong.")
        return AIMessage(content="YES")

fake_llm = GenericFakeChatModel(messages=MessagesIterator())

class JobApplicationState(TypedDict):
    job_description: str
    is_suitable: bool
    application: str
    actions: Annotated[list[str], add]

def generate_application(state):
    print("...generating application...")
    return {"application": "some_fake_application", "actions": ["action2"]}

def is_suitable_condition(state: JobApplicationState) -> Literal["generate_application", END]:
    if state.get("is_suitable"):
        return "generate_application"
    return END

def analyze_job_description(state, config: RunnableConfig):
    try:
        print("here")
        llm = config["configurable"].get("model_provider", "Google")
        analyze_chain = llm | parser
        prompt = prompt_template_enum.format(job_description=job_description)
        result = analyze_chain.invoke(prompt)
        return {"is_suitable": result}
    except Exception as e:
        logger.error(f"Exception {e} occured while executing analyze_job_description")
        return {"is_suitable": False}

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description)
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

show_graph(graph)

# 这里调用的模型提供商是fake，运行analyze_job_description方法的第92行时，会报错。但是在except...代码块中，进行兜底，最终返回{“is_suitable": False}
res = graph.invoke({"job_description": "fake_jd"}, config={"configurable": {"model_provider": "fake"}})
print(res)

# 3. 带有重试机制的chain
analyze_chain_fake = fake_llm | parser

fake_llm_retry = fake_llm.with_retry(
    retry_if_exception_type=(ValueError,),
    wait_exponential_jitter=True,
    stop_after_attempt=2,
)

analyze_chain_fake_retries = (fake_llm | parser).with_retry(
    retry_if_exception_type=(ValueError,),
    wait_exponential_jitter=True,
    stop_after_attempt=2,
)

# 这里第一次调用fake_llm时，会抛出ValueError的异常。但是经过重试，第二次调用fake_llm时，会返回IsSuitableJobEnum.YES: "YES"
print(analyze_chain_fake_retries.invoke("test"))


llms = {
    "fake": fake_llm,
    "Google": model
}

# 4. 在builder.add_note方法中，使用retry_policy添加重试机制
def analyze_job_description(state, config: RunnableConfig):
    model_provider = config["configurable"].get("model_provider", "Google")
    llm = llms[model_provider]
    analyze_chain = llm | parser
    prompt = prompt_template_enum.format(job_description=job_description)
    result = analyze_chain.invoke(prompt)
    return {"is_suitable": result}

builder = StateGraph(JobApplicationState)
builder.add_node("analyze_job_description", analyze_job_description, retry_policy=RetryPolicy(retry_on=ValueError, max_attempts=2))
builder.add_node("generate_application", generate_application)
builder.add_edge(START, "analyze_job_description")
builder.add_conditional_edges("analyze_job_description", is_suitable_condition)
builder.add_edge("generate_application", END)

graph = builder.compile()

show_graph(graph)

res = graph.invoke({"job_description": job_description}, config={"configurable": {"model_provider": "fake"}})
print(res)

res = graph.invoke({"job_description": "fake_jd"}, config={"configurable": {"model_provider": "fake"}})
print(res)

response = fake_llm_retry.invoke("test")
print(response)


# 5. 回滚机制，第一次执行chain_with_fb时会报错，执行回滚，输出running fallback;第二次执行chain_with_fb时，无报错，输出running main chain
chain_fallback = RunnableLambda(lambda _: print("running fallback"))
chain = fake_llm | RunnableLambda(lambda _: print("running main chain"))
chain_with_fb = chain.with_fallbacks([chain_fallback])

chain_with_fb.invoke("test")
chain_with_fb.invoke("test")

