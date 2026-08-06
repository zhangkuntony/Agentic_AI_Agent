import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter


model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
)

math_cot_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a math problem-solving assistant. "
            "Please think step by step to solve the problem "
            "and provide the final answer.",
        ),
        ("human", "{input}"),
    ]
)
cot_chain = math_cot_prompt | model | StrOutputParser()
# print(cot_chain.invoke("Solve equation 2*x+5=15"))

parse_prompt_template = (
    "Given the initial question and a full answer, "
    "extract the concise answer. Do not assume anything and "
    "only use a provided full answer. \n\nQUESTION:\n{question}\n"
    "FULL ANSWER:\n{full_answer}\n\nCONCISE ANSWER:\n"
)
parse_prompt = PromptTemplate.from_template(parse_prompt_template)
final_chain = (
    {"full_answer": itemgetter("question") | cot_chain, "question": itemgetter("question"),}
    | parse_prompt | model | StrOutputParser()
)

# print(final_chain.invoke({"question": "Solve equation 2*x**2-96*x+1152"}))

generations = []
for i in range(20):
    result = final_chain.invoke({"question": "Solve equation 2*x**2-96*x+1152"}, temperature=2.0)
    print(f"{i}: {result}")
    generations.append(result.strip())

from collections import Counter
print("Print most common answers:")
print(Counter(generations).most_common(1)[0][0])