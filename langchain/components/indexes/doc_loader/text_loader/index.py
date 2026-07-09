from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader


load_dotenv()


model = ChatOpenAI()
parser = StrOutputParser()

prompt = PromptTemplate.from_template(
    "summarize the text in 5 lines \n {text}"
)


loader = TextLoader("langchain.txt", encoding="utf-8")
docs = loader.load()


chain = prompt | model | parser

response = chain.invoke({"text": docs[0].page_content})
print(response)

