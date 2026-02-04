from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch,RunnableLambda

llm = ChatOllama(
    model="qwen2:7b",
    temperature=0   
)

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )

parser = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback.\n\n"
        "{format_instructions}\n\n"
        "Feedback:\n{feedback}"
    ),
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

classifier_chain = prompt | llm | parser

prompt2 = PromptTemplate(
    template='write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3 = PromptTemplate(
    template='write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive',prompt2 | llm | parser),
    (lambda x:x.sentiment == 'negative',prompt2 | llm | parser),
    RunnableLambda(lambda x :'could not find sentiment')
)

chain = classifier_chain | branch_chain 

result = chain.invoke({'feedback':'This Laptop is not Working very well'})

print(result)

chain.get_graph().print_ascii()

# result = classifier_chain.invoke(
#     {"feedback": "Service was so slow and bad"}
# )

# print(result.sentiment)
