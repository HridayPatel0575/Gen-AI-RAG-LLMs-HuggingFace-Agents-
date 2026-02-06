from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough


llm = ChatOllama(
    model='qwen2:7b',
    temperature=0
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain this joke /n {text}',
    input_variables=['text']
)

joke_chain = RunnableSequence(prompt,llm,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,llm,parser)
})


joint_chain = joke_chain | parallel_chain

print(joint_chain.invoke({'topic':'Dog'}))