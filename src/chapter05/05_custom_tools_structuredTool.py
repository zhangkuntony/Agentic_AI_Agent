from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.tools import StructuredTool, ToolException
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field

import math
import numexpr as ne

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

# class CalculatorArgs(BaseModel):
#     expression: str = Field(description="Mathematical expression to be evaluated")
#
# def calculator(expression: str) -> str:
#     math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
#     result = ne.evaluate(expression.strip(), local_dict=math_constants)
#     return str(result)
#
# calculator_tool = StructuredTool.from_function(
#     name="calculator",
#     description=("Calculates a single mathematical expression, incl. complex numbers."),
#     func=calculator,
#     args_schema=CalculatorArgs
# )
#
# llm_with_tools = llm.bind_tools([calculator_tool])
# tool_call = llm_with_tools.invoke("How much is (2+3i)**2").tool_calls[0]
# print(tool_call)

def calculator(expression: str) -> str:
    """Calculates a single mathematical expression, incl. complex numbers."""
    math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
    try:
        result = ne.evaluate(expression.strip(), local_dict=math_constants)
    except Exception as e:
        # 关键：把普通异常包装成 ToolException，
        # handle_tool_error=True 才会把它转成 ToolMessage 回传给模型
        raise ToolException(f"Error: {e!r}\n Please fix your mistakes.") from e
    return str(result)

calculator_tool = StructuredTool.from_function(
    func=calculator,
    handle_tool_error=True
)

agent = create_agent(llm, [calculator_tool])

for event in agent.stream({"messages": [("user", "How much is (2+3i)^2")]}, stream_mode="values"):
    event["messages"][-1].pretty_print()
