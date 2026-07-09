from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import  DirectoryLoader, PyPDFLoader

load_dotenv()


model = ChatOpenAI()
parser = StrOutputParser()


directory_loader = DirectoryLoader(
    path="books",
    glob="*.pdf", # used to set the format of files to load
    loader_cls=PyPDFLoader
)

docs = directory_loader.load()
