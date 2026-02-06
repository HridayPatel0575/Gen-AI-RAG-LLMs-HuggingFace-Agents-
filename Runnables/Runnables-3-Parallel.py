from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel


llm = ChatOllama(
    model='qwen2:7b',
    temperature=0
)


prompt = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate Linkedin post about {topic}',
    input_variables=['topic']
)


parser = StrOutputParser()

chain = RunnableParallel({
    'tweet':RunnableSequence(prompt,llm,parser),
    'linkedin':RunnableSequence(prompt2,llm,parser)
})



print(chain.invoke({'topic':'AI'}))