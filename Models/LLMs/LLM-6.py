import torch
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

hf_pipeline = pipeline(
    task="text-generation",
    model="meta-llama/Llama-3.1-8B",
    device_map="auto",            
    torch_dtype=torch.float16,    
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

chat = ChatHuggingFace(llm=llm)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = chat.invoke(messages)

print(outputs.content)
