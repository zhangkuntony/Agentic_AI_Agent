import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import API_KEY, BASE_URL, MODEL_NAME
from langchain_classic.output_parsers import RetryWithErrorOutputParser, PydanticOutputParser
from langchain_classic.output_parsers.retry import NAIVE_RETRY_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

class SearchAction(BaseModel):
    query: str = Field(description="A query to search for if a search action is taken")

parser = PydanticOutputParser(pydantic_object=SearchAction)

# 这里故意写一个错误的json字符串。正确的写法应该是： completion_with_error = '{"query": "what is the weather like in Munich tomorrow"}'
completion_with_error = "{'action': 'what is the weather like in Munich tomorrow}"

# 这里执行的时候一定会抛出异常
try:
    response = parser.parse(completion_with_error)
    print(response)
except Exception as e:
    print(e)

# 创建带修复能力的解析器
fix_parser = RetryWithErrorOutputParser.from_llm(
    llm=model,
    parser=parser,
)

# 查看langchain内置的“重试提示词模板”
print(NAIVE_RETRY_PROMPT)

# 自定义修复提示词模板
retry_template = (
    "Your previous response doesn't follow the required schema and fails parsing. Fix the response so that it follow the expected schema."
    "Do not change the nature of response, only adjust the schema."
    "\n\nEXPECTED SCHEMA:{schema}\n\n"
)
retry_prompt = PromptTemplate.from_template(retry_template)

# 执行修复+解析
fixed_output = fix_parser.parse_with_prompt(
    completion=completion_with_error,
    prompt_value=retry_prompt.format_prompt(schema=parser.get_format_instructions())
)
print(fixed_output)


# 在模型上绑定结构化输出，从源头上避免解析失败
structured_model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
).with_structured_output(SearchAction)

result = structured_model.invoke("What is the weather like in Munich tomorrow?")
print(result)
print(result.query)