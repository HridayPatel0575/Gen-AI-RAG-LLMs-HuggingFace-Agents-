from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import torch


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    # device_map="auto",
    # torch_dtype="auto"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke('Delhi is capital of India')

print(result)

