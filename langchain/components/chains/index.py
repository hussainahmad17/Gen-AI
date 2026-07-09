from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Prompt 1: Generate a detailed report
prompt1 = PromptTemplate.from_template(
    "Generate a detailed report on {topic}."
)

# Prompt 2: Summarize the report
prompt2 = PromptTemplate.from_template(
    "Generate a 5-point summary from the following text:\n\n{text}"
)

# LLM
model = ChatOpenAI()

# Output parser
parser = StrOutputParser()

# Sequential Chain
chain = prompt1 | model | parser | prompt2 | model | parser

# Invoke the chain
result = chain.invoke({
    "topic": "Unemployment"
})

print(result)