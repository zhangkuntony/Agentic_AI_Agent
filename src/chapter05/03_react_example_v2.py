import math

from langchain_core.messages import HumanMessage

from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

def mocked_google_search(query: str) -> str:
    print(f"CALLED GOOGLE SEARCH with query={query}")
    return (
        "The current US President (as of 2025) is Donald Trump. "
        "He was born on June 14, 1946, making him 79 years old in 2025. "
        "His age is 79."
    )

def mocked_calculator(expression: str) -> float:
    print(f"CALLED CALCULATOR with expression={expression}")
    expr = expression.replace("sqrt", "math.sqrt")
    return eval(expr, {"math": math})

system_prompt = (
    "Always use a calculator for mathematical computations, and use Google Search "
    "for information about common facts, fresh events and news. Do not assume anything, keep in "
    "mind that things are changing and always "
    "check yourself with external sources if possible."
)

@tool("google_search")
def mocked_google_search_tool(query: str) -> str:
    """Returns about common facts, fresh events and news from Google Search engine based on a query."""
    return mocked_google_search(query)

@tool("calculator")
def mocked_calculator_tool(expression: str) -> str:
    """Computes mathematical expressions"""
    return str(mocked_calculator(expression))

agent = create_agent(
    model=llm,
    tools=[mocked_google_search_tool, mocked_calculator_tool],
    system_prompt=system_prompt
)

question = "What is a square root of (the current US president's age multiplied by 132)?"
agent_result = agent.invoke({"messages": [HumanMessage(content=question)]})
print(agent_result["messages"][-1].content)
