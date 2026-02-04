from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Initialize Ollama chat model
llm = ChatOllama(
    model="qwen2:7b",
    temperature=0.7
)

# Prompt templates
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary of the following text:\n{text}",
    input_variables=["text"]
)

prompt1 = template1.format(topic="India")
response1 = llm.invoke(
    [HumanMessage(content=prompt1)]
)

detailed_report = response1.content

prompt2 = template2.format(text=detailed_report)
response2 = llm.invoke(
    [HumanMessage(content=prompt2)]
)

summary = response2.content

print(summary)
