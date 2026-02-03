from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage


llm = ChatOllama(
    model="qwen2:7b",
    temperature=0.7
)

response = llm.invoke([
    HumanMessage(content="Explain transformers in simple words")
])

print(response.content)

