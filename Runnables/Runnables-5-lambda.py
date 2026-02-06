from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda


llm = ChatOllama(
    model='qwen2:7b',
    temperature=0
)

parser = StrOutputParser()

def word_count(text):
    
    return len(text.split())

prompt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)


joke_chain = RunnableSequence(prompt, llm, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = joke_chain | parallel_chain

result = final_chain.invoke({'topic':'Cinema'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)