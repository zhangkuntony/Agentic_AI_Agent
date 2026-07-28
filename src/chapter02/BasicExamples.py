import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.language_models import FakeListLLM
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. LLM 交互模式
model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

response = model.invoke("Tell me a joke about light bulbs!")
print(response.content)

# 2. 使用FakeListLLM进行开发测试

fake_llm = FakeListLLM(responses=["Hello"])     # 创建一个始终返回相同响应的虚拟LLM

result = fake_llm.invoke("Any input will return Hello")
print(result)

# 3. 使用聊天模型

messages = [
    SystemMessage(content="You're a helpful programming assistant"),
    HumanMessage(content="Write a Python function to calculate factorial")
]
response = model.invoke(messages)
print(response.content)

# 4. 推理模型

# 创建一个模板
template = ChatPromptTemplate.from_messages([
    ("system", "You are an experienced programmer and mathematical analyst."),
    ("user", "{problem}")
])

# 使用 reasoning_effort 参数初始化
chat = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    reasoning_effort="high"     # 选项: "low", "medium", "high"
)

# 创建并运行一个链
chain = template | chat

# 复杂的算法问题
problem = """
Design an algorithm to find the kth largest element in an unsorted array
with the optimal time complexity. Analyze the time and space complexity
of your solution and explain why it's optimal.
"""

response = chain.invoke({"problem": problem})
print(response)
