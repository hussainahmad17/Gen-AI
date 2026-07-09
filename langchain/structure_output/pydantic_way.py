from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field # in pydantic, you can validate the data
from dotenv import load_dotenv
load_dotenv()


model = ChatOpenAI(model_name='gpt-4o')


# defining schema by creating a class
class Review(BaseModel):

    summary: str = Field(description="Brief summary of review")
    sentiment: str = Field(description="Return the sentiment of review, either positive, neutral, or negative")
    key_points: str = Field(description= "Return main features of phone in list format")


structured_model =  model.with_structured_output(Review)
response = structured_model.invoke('Ive been using the iPhone 17 Pro Max for a couple of weeks, and its been impressive. The battery easily lasts all day, the camera takes sharp photos even in low light, and everything feels incredibly smooth. Its expensive, but if you want a premium phone with excellent performance, its definitely worth considering.')

print(response)