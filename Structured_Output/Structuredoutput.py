from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

llm = ChatOllama(
    model="qwen2:7b",
    temperature=0.7
)

class Facts(BaseModel):
    fact_1: str = Field(description="Fact 1 about the topic")
    fact_2: str = Field(description="Fact 2 about the topic")
    fact_3: str = Field(description="Fact 3 about the topic")

parser = PydanticOutputParser(pydantic_object=Facts)

prompt = PromptTemplate(
    template=(
        "Give 3 facts about {topic}.\n"
        "Respond ONLY in JSON.\n"
        "{format_instructions}"
    ),
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = prompt | llm | parser

result = chain.invoke({"topic": "black hole"})

print(result)
print(result.fact_1)
