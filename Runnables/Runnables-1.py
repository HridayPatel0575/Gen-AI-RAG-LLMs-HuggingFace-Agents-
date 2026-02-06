from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model='qwen2:7b',temperature=0
)

template = PromptTemplate(template='Write 5 lines on {topic}',
                          input_variables=['topic'])

topic = 'india'

prompt = template.format(topic = topic)

respond = llm.invoke(
    [HumanMessage(content=prompt)]
)

print(respond.content)
# template = PromptTemplate(
#      template="write one line on {topic}",
#      input_variables=['topic']
# )

# prompt = template.format(topic='India')

# response = llm.invoke(
#     [HumanMessage(content = prompt)]
#     )


# print(response)

# parser = StrOutputParser()

# chain = prompt | llm | parser

# result = chain.invoke({})
