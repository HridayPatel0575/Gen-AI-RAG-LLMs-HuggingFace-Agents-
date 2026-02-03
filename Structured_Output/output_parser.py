from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    device=-1
    # max_new_tokens=500,
    # temperature=0.7
)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary of the following text:\n{text}",
    input_variables=["text"]
)

prompt1 = template1.format(topic="India")
result = llm.invoke(prompt1)

prompt2 = template2.format(text=result)
result1 = llm.invoke(prompt2)

print(result1)