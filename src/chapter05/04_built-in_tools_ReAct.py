from config.config import API_KEY, BASE_URL, MODEL_NAME
from datetime import date
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from utils.utils import show_graph

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

_search = DuckDuckGoSearchRun(api_wrapper_kwargs={"backend": "html"})

@tool
def duckduckgo_search(query: str) -> str:
    """Search real time news by DuckDuckGo"""
    try:
        return _search.invoke({"query": query})
    except Exception as e:
        return (f"Search temporarily failed ({type(e).__name__}: {e}). "
                f"Do NOT retry again; answer with what you know or state the limitation.")

today = date.today().isoformat()

prompt = f"""
Today is {today}. The user's "tomorrow" means {today} followed by one day,
so always convert it into a concrete date and never guess the year/month.

When you use the search tool:
- Search at most once or twice.
- Search snippets often lack exact figures. Extract whatever facts exist;
  if no concrete number is available, give a qualitative answer and clearly
  state that exact data is unavailable.
- Never rephrase the query and search repeatedly.
"""

agent = create_agent(
    model=llm,
    tools=[duckduckgo_search],
    system_prompt=prompt
)

show_graph(agent)

query = "What is the weather in Shanghai like tomorrow?"

for event in agent.stream(
    {"messages": [("user", query)]},
    config={"recursion_limit": 10},
):
    for key, value in event.items():
        for m in value.get("messages", []):
            m.pretty_print()