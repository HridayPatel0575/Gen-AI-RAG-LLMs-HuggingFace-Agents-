from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text = "Modi is PM of India"
vector = embeddings.embed_query(text)

print(len(vector))
print(vector[:10]) 
