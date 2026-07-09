from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
model_response = model.invoke("PM of Pakistan?")
print(model_response.content)