from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, convert_runnable_to_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import math
import numexpr as ne

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

def calculator(expression: str) -> str:
    math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
    result = ne.evaluate(expression.strip(), local_dict=math_constants)
    return str(result)

calculator_with_retry = RunnableLambda(calculator).with_retry(
    wait_exponential_jitter=True,
    stop_after_attempt=3
)

calculator_tool = convert_runnable_to_tool(
    calculator_with_retry,
    name="calculator",
    description=(
        """
        Calculates a single mathematical expression, incl. complex numbers.

        Rules (Python syntax):
          - Use ** for exponentiation, NEVER ^ (e.g. (2+3*i)**2)
          - Always add * between a number and a constant/variable, examples:
              73i -> 73*i
              7pi**2 -> 7*pi**2
          - Available constants: pi, e, i (imaginary unit = 1j)
        """
    ),
    arg_types={"expression": str},
)

print(calculator_tool.invoke({"expression": "(2+3*i)**2"}))

llm_with_tools = llm.bind_tools([calculator_tool])

print(llm_with_tools.invoke("How much is (2+3i)**2").tool_calls[0])


# We can also pass a custom arguments schema when we create a tool with `convert_runnable_to_tool`
class CalculatorArgs(BaseModel):
    expression: str = Field(description="Mathematical expression to be evaluated")

def calculator(state: CalculatorArgs, config: RunnableConfig) -> str:
    expression = state["expression"]
    math_constants = config["configurable"].get("math_constants", {})
    result = ne.evaluate(expression.strip(), local_dict=math_constants)
    return str(result)

calculator_with_retry = RunnableLambda(calculator).with_retry(
    wait_exponential_jitter=True,
    stop_after_attempt=3
)

calculator_tool = convert_runnable_to_tool(
    calculator_with_retry,
    name="calculator",
    description=(
        """
        Calculates a single mathematical expression, incl. complex numbers.
        
        Rules (Python syntax):
          - Use ** for exponentiation, NEVER ^ (e.g. (2+3*i)**2)
          - Always add * between a number and a constant/variable, examples:
              73i -> 73*i
              7pi**2 -> 7*pi**2
          - Available constants: pi, e, i (imaginary unit = 1j)
        """
    ),
    args_schema=CalculatorArgs,
    arg_types={"expression": str},
)

assert isinstance(calculator_tool, BaseTool)
print(f"Tool name: {calculator_tool.name}")
print(f"Tool description: {calculator_tool.description}")
print(f"Args schema: {calculator_tool.args_schema.model_json_schema()}")

math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
config = {"configurable": {"math_constants": math_constants}}

llm_with_tools = llm.bind_tools([calculator_tool])
tool_call = llm_with_tools.invoke("How much is (2+3i)**2").tool_calls[0]

print(tool_call)

print(calculator_tool.invoke(tool_call["args"], config=config))