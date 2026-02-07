from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.underarmour.com/en-us/'
loader = WebBaseLoader(url)

docs = loader.load()

print(docs)