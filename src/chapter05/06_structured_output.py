from config.config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import json

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME
)

# 1. Use Pydantic Model structed output

class Step(BaseModel):
    """A step that is a part of the plan to solve the task."""
    step: str = Field(description="Description of the step")

class Plan(BaseModel):
    """A plan to solve the task"""
    steps: list[Step]

prompt = PromptTemplate.from_template(
    "Prepare a step-by-step plan to solve the given task.\n"
    "TASK:\n{task}\n"
)

result = (prompt | llm.with_structured_output(Plan)).invoke("How to write a bestseller on Amazon about generative AI?")
assert isinstance(result, Plan)
print(f"Amount of steps: {len(result.steps)}")
for step in result.steps:
    print(step.step)

# 2. Use json_mode and pass a custom schema to LLM for output

plan_schema = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"step": {"type": "string"}},
                "required": ["step"],
            },
        }
    },
    "required": ["steps"],
}

prompt = PromptTemplate.from_template(
    "Prepare a step-by-step plan to solve the given task.\n"
    "Respond ONLY with JSON matching this schema:\n"
    "{schema}\n"
    "TASK:\n{task}\n"
).partial(schema=json.dumps(plan_schema))

query = "How to write a bestseller on Amazon about generative AI?"
result = (prompt | llm.with_structured_output(schema=plan_schema, method="json_mode")).invoke(query)

assert(isinstance(result['steps'], list))
print(f"Amount of steps: {len(result['steps'])}")
print(result["steps"])


# 3. Use custom arguments supported by the LLM provider

plan_schema = {
    "name": "plan_schema",
    "description": "Creates a List of steps to achieve a goal",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "List of steps to achieve the goal",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {
                            "type": "string",
                            "description": "A step to achieve part of the goal",
                        }
                    },
                    "additionalProperties": False,
                    "required": ["step"],
                },
            }
        },
        "additionalProperties": False,
        "required": ["steps"],
    },
}

response_format = {"type": "json_schema", "json_schema": plan_schema}

llm_json = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    model_kwargs={"response_format": response_format}
)

query = "How to write a bestseller on Amazon about generative AI?"
prompt = PromptTemplate.from_template(
    "Prepare a step-by-step plan to solve the given task.\n"
    "TASK:\n{task}\n"
)

result = (prompt | llm_json | JsonOutputParser()).invoke(query)
assert isinstance(Plan.model_validate(result), Plan)
print(f"Amount of steps: {len(result['steps'])}")
print(result["steps"])


# 4. Generate an enum output

response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_classifier",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                }
            },
            "required": ["sentiment"],
            "additionalProperties": False,
        },
    },
}

prompt = PromptTemplate.from_template(
    "Classify the tone of the following customer's review" "\n{review}\n"
)

review = "I like this movie!"
llm_enum = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    model_kwargs={"response_format": response_schema}
)
result = (prompt | llm_enum | JsonOutputParser()).invoke(review)
print(result)
print(result["sentiment"])