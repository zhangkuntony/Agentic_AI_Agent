import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import API_KEY, BASE_URL, MODEL_NAME
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from operator import itemgetter

model = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME
)

# 1. Langchain Common Expression Language (LCEL)
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
output_parser = StrOutputParser()

# Chain them together using LCEL
chain = prompt | model | output_parser

# Use the chain
result = chain.invoke({"topic": "programming"})
print(result)


# 2. More complex expressions

# Fist chain generates a story
story_prompt = PromptTemplate.from_template("Write a short story about {topic}")
story_chain = story_prompt | model | StrOutputParser()

# Second chain analyzes the story
analysis_prompt = PromptTemplate.from_template("Analyze the following story's mood:\n{story}")
analysis_chain = analysis_prompt | model | StrOutputParser()

# Combine chains
story_with_analysis = story_chain | analysis_chain

# Run the combined chain
story_analysis = story_with_analysis.invoke({"topic": "a rainy day"})
print("\nAnalysis:", story_analysis)


# 3. 使用RunnablePassthrough.assign 保留数据
enhanced_chain = RunnablePassthrough.assign(
    story=story_chain           # 添加 'story' 键，值为生成的内容
).assign(
    analysis=analysis_chain     # 添加 'analysis' 键，值为对故事的分析
)
# 执行链
result = enhanced_chain.invoke({"topic": "a rainy day"})
print(f"Result.keys(): {result.keys()}")
print(f"Result.topic: {result['topic']}")
print(f"Result.story: {result['story']}")
print(f"Result.analysis: {result['analysis']}")


# 4. 使用字典构造的替代方法
manual_chain = (
    RunnablePassthrough() |                 # 传递输入
    {
        "story": story_chain,               # 添加故事结果
        "topic": itemgetter("topic")        # 保留原始主题
    } |
    RunnablePassthrough().assign(           # 根据故事添加分析
        analysis=analysis_chain,
    )
)
manual_chain_result = manual_chain.invoke({"topic": "a rainy day"})
print(f"Result.keys(): {manual_chain_result.keys()}")
print(f"Result.topic: {manual_chain_result['topic']}")
print(f"Result.story: {manual_chain_result['story']}")
print(f"Result.analysis: {manual_chain_result['analysis']}")


# 5. 简化的字典构造
simple_dict_chain_corrected = story_chain | {
    "story": RunnablePassthrough(),         # 将故事输出作为 'story' 传递
    "analysis": analysis_chain
}

# analysis_chain 将按照预期接收 {'story': '实际故事内容'}.
result_corrected = simple_dict_chain_corrected.invoke({"topic": "a rainy day"})
print(f"Result.keys(): {result_corrected.keys()}")
print(f"Result.story: {result_corrected['story']}")
print(f"Result.analysis: {result_corrected['analysis']}")