from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME
)
question = "how old is the US president?"

search_tool = {
    "type": "function",
    "function": {
        "name": "google_search",
        "description": "Returns about common facts, fresh events and news from Google Search engine based on a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "title": "search_query",
                    "description": "Search query to be sent to the search engine"
                }
            },
            "required": ["query"]
        }
    }
}
step1 = llm.invoke(question, tools=[search_tool])
print(step1.tool_calls)

tool_result = ToolMessage(content="Donald Trump > Age 78 years June 14, 1946\n", tool_call_id=step1.tool_calls[0]["id"])
step2 = llm.invoke(
    [HumanMessage(content=question), step1, tool_result],
    tools=[search_tool]
)
assert len(step2.tool_calls) == 0

print(step2.content)

llm_with_tools = llm.bind(tools=[search_tool])
response = llm_with_tools.invoke(question)
print(response.tool_calls)

