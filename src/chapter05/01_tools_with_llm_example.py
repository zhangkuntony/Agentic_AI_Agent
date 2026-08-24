from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME
)

question = "how old is the US president?"

raw_prompt_template = (
    "You have access to search engine that provides you an "
    "information about fresh events and news given the query. "
    "Given the question, decide whether you need an additional "
    "information from the search engine (reply with 'SEARCH: "
    "<generated query>' or you know enough to answer the user "
    "then reply with 'RESPONSE <final response>').\n"
    "Do not make any assumptions on recent events or things that can change."
    "Now, act to answer a user question:\n{QUESTION}"
)
prompt_template = PromptTemplate.from_template(raw_prompt_template)

question1 = "what is the capital of Germany?"
result = (prompt_template | llm).invoke(question1)
print(result.content)

response = (prompt_template | llm).invoke(question)
print(response.content)

query = "age of current US president"
# Will print correct age of current US president by using below search result
search_result = (
    "Donald Trump › Age 78 years June 14, 1946\n"
    "Donald Trump 45th and 47th U.S. President Donald John Trump is an American "
    "politician, media personality, and businessman who has served as the 47th "
    "president of the United States since January 20, 2025. A member of the "
    "Republican Party, he previously served as the 45th president from 2017 to 2021. Wikipedia"
)

## Will print "SEARCH: Donald Trump age 2026" by using below search result
# search_result = (
#     "Donald Trump 45th and 47th U.S."
# )

raw_prompt_template = (
    "You have access to search engine that provides you an "
    "information about fresh events and news given the query. "
    "Given the question, decide whether you need an additional "
    "information from the search engine (reply with 'SEARCH: "
    "<generated query>' or you know enough to answer the user "
    "then reply with 'RESPONSE <final response>').\n"
    "Today is {date}."
    "Now, act to answer a user question and "
    "take into account your previous actions:\n"
    "HUMAN: {question}\n"
    "AI: SEARCH: {query}\n"
    "RESPONSE FROM SEARCH: {search_result}\n"
)
prompt_template = PromptTemplate.from_template(raw_prompt_template)

result = (prompt_template | llm).invoke({"question": question, "query": query, "search_result": search_result, "date": "Aug 2026"})
print(result.content)