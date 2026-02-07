from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


llm = ChatOllama(
    model='qwen2:7b',
    temperature=0
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt',encoding='utf-8')

docs = loader.load()

prompt = PromptTemplate(
    template='Write summary for {text}',
    input_variables=['text']
    )

chain = prompt | llm | parser

response=chain.invoke({'text':docs[0].page_content})

print(response)