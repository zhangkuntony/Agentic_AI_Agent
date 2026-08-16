from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME
)

# 1. LLM 提示模板

template = """
Summarize this text in one sentence:

{text}
"""

prompt = PromptTemplate.from_template(template)

# Format the prompt with actual values
formatted_prompt = prompt.format(text="""
Explanation:
1. The function takes a single parameter `n`, which represents the number for which we want to calculate the factorial.

2. We start by checking if `n` is negative. If it is, we raise a `ValueError` because factorial is not defined for negative numbers.

3. Next, we check if `n` is equal to 0 or 1. If it is, we return 1 since the factorial of 0 and 1 is defined as 1.

4. If `n` is greater than 1, we initialize a variable `result` to 1. This variable will store the factorial value.

5. We use a `for` loop to iterate from 2 to `n` (inclusive). In each iteration, we multiply `result` by the current value of `i`. This effectively calculates the factorial by multiplying all the numbers from 2 to `n`.

6. Finally, we return the calculated factorial value stored in `result`.
""")

result = model.invoke(formatted_prompt)
print(result.content)


# 2. 聊天提示模板
template = ChatPromptTemplate.from_messages([
    ("system", "You are an English to French translator"),
    ("user", "Translate this to French: {text}")
])

formatted_prompt = template.format_messages(text="Hello, how are you?")
result = model.invoke(formatted_prompt)
print(result.content)