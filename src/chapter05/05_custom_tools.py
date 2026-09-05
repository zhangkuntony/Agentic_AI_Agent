from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

import math
import numexpr as ne

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
print(ne.evaluate("(2+3*i)**2", local_dict=math_constants))

@tool
def calculator(expression: str) -> str:
    """Calculates a single mathematical expression, incl. complex numbers.

    Rules (Python syntax):
      - Use ** for exponentiation, NEVER ^ (e.g. (2+3*i)**2)
      - Always add * between a number and a constant/variable, examples:
          73i -> 73*i
          7pi**2 -> 7*pi**2
      - Available constants: pi, e, i (imaginary unit = 1j)
    """
    math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
    result = ne.evaluate(expression.strip(), local_dict=math_constants)
    return str(result)

assert isinstance(calculator, BaseTool)
print(f"Tool name: {calculator.name}")
print(f"Tool description: {calculator.description}")
print(f"Tool schema: {calculator.args_schema.model_json_schema()}")

query = "How much is 2+3i squared?"

agent = create_agent(llm, [calculator])

for event in agent.stream({"messages": [("user", query)]}, stream_mode="values"):
    event["messages"][-1].pretty_print()

search = DuckDuckGoSearchRun()

question = "What is a square root of the current US president’s age multiplied by 132?"

system_hint = "Think step-by-step. Always use search to get the fresh information about events or public facts that can change over time. Now is 2025 and remember president elections in the US recently happened."

agent = create_agent(llm, [calculator, search], system_prompt=system_hint)

for event in agent.stream({"messages": [("user", question)]}, stream_mode="values"):
    event["messages"][-1].pretty_print()

