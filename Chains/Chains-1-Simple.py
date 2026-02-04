from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatOllama(
    model='qwen2:7b',
    temperature=0.7
)

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | llm | parser

# result = chain.invoke({'topic':'Football'})

# print(result)

chain.get_graph().print_ascii()