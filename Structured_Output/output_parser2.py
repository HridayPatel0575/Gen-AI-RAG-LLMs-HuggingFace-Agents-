from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",   
    max_new_tokens=512,
    temperature=0.7
)

chat_model = ChatHuggingFace(llm=llm)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Write a detailed report on black holes.")
])

result = chat_model.invoke(prompt.invoke({}))


print(result.content)
