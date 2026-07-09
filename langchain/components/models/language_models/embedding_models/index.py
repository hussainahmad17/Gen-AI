from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
embedding = embedding_model.embed_query("What is the capital of Pakistan?")
print(str(embedding))