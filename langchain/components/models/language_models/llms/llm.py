from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAI



llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
result = llm.invoke("PM of Pakistan?")
print(result)