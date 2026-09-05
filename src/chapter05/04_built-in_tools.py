from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.ddg_search.tool import DDGInput
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

search = DuckDuckGoSearchRun(api_wrapper_kwargs={"backend": "api"})
print(f"Tool's name = {search.name}")
print(f"Tool's description = {search.description}")
print(f"Tool's arg schema = {search.args_schema}")

print(DDGInput.model_fields)

query = "What is the weather in Shanghai like tomorrow?"
search_input = DDGInput(query=query)
result = search.invoke(search_input.model_dump())
print(result)

llm_with_tools = llm.bind_tools([search])
result = llm_with_tools.invoke(
    [
        ("system", "Always use a duckduckgo_search tool for queries that require a fresh information"),
        ("user", query)
    ]
)
print(result.tool_calls[0])

result = llm_with_tools.invoke(
    [
        ("system", "Always use a duckduckgo_search tool for queries that require a fresh information"),
        ("user", "How much is 2+2?")
    ]
)
assert not result.tool_calls

