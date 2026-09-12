from config.config import API_KEY, BASE_URL, MODEL_NAME
from datetime import date, timedelta
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

import math
import numexpr as ne

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME
)

search = DuckDuckGoSearchRun()

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

llm_with_tools = llm.bind_tools([search, calculator])

def invoke_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("invoke_llm", invoke_llm)
builder.add_node("tools", ToolNode([search, calculator]))

builder.add_edge(START, "invoke_llm")
builder.add_conditional_edges("invoke_llm", tools_condition)
builder.add_edge("tools", "invoke_llm")
builder.add_edge("tools", END)
builder.add_edge("invoke_llm", END)
graph = builder.compile()

for e in graph.stream({"messages": ("human", "How much is 2+2")}):
    print(e)

examples = [
    "I signed my contract 2 years ago",
    "I started the deal with your company in February last year",
    "Our contract started on March 24th two years ago"
]

@tool
def get_date(year: int, month: int = 1, day: int = 1) -> date:
    """Returns a date object given year, month and day.

      Default month and day are 1 (January) and 1.
      Examples in YYYY-MM-DD format:
        2023-07-27 -> date(2023, 7, 27)
        2022-12-15 -> date(2022, 12, 15)
        March 2022 -> date(2022, 3)
        2021 -> date(2021)
    """
    return date(year, month, day).isoformat()

@tool
def time_difference(days: int = 0, weeks: int = 0, months: int = 0, years: int = 0) -> date:
    """Returns a date given a difference in days, weeks, months and years relative to the current date.

    By default, days, weeks, months and years are 0.
    Examples:
      two weeks ago -> time_difference(weeks=2)
      last year -> time_difference(years=1)
    """
    dt = date.today() - timedelta(days=days, weeks=weeks)
    new_year = dt.year + (dt.month - months) // 12 - years
    new_month = (dt.month - months) % 12
    return dt.replace(year=new_year, month=new_month)

agent = create_agent(
    llm, [get_date, time_difference], system_prompt="Extract the starting date of a contract. Current year is 2026."
)

for example in examples:
    result = agent.invoke({"messages": [("user", example)]})
    print(example, result["messages"][-1].content)

for e in agent.stream({"messages": [("user", examples[1])]}):
    print(e)

print(result["messages"][-1].content)