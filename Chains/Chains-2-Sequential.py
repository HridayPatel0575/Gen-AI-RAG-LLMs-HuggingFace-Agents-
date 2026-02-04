from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
    model='qwen2:7b',
    temperature=0.7
)

prompt1 = PromptTemplate(
    template='Generate Detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate 5 line summary on {text}',
    input_variables=['text']
)


parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser 

result = chain.invoke({'topic':'Cricket'})
print(result)

chain.get_graph().print_ascii()