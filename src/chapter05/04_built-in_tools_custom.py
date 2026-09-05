from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

python_repl = PythonREPL()
python_repl.run("print(2**4)")

code_interpreter_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

query_strawberry = "How many r are in the word strawberry?"
print(llm.invoke(query_strawberry).content)

agent = create_agent(
    model=llm,
    tools=[code_interpreter_tool],
)

for event in agent.stream(
    {"messages": [("user", query_strawberry)]},
    config={"recursion_limit": 10},
):
    for key, value in event.items():
        for m in value.get("messages", []):
            m.pretty_print()