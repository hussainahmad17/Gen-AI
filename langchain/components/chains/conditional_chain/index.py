from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()


first_model = ChatOpenAI(model="gpt-4o")

class Review(BaseModel):
    sentiment: str = Field(description="Return the sentiment of review, either positive, neutral, or negative")

structured_model = first_model.with_structured_output(Review)

msg_prompt = PromptTemplate.from_template(
    """
Write the message for the customer, depending upon its review. If review is positive, write positve feedback msg, if negative then write so.
Customer review: {customer_review}
Sentiment: {sentiment}
"""
)


parser = StrOutputParser()

review = "I bought this laptop last month for remote work and light gaming, but honestly, the cheap plastic chassis immediately disappointed me because it flexes and creaks whenever I open the lid. The screen is incredibly dim and washed out, making it practically impossible to read text if I am sitting anywhere near a window during the day. Typing on the keyboard feels horribly mushy and unresponsive, and the spacebar already sticks half the time after just a few weeks of use. It weighs way more than advertised, causing immediate shoulder pain when I try to carry it in my backpack during my morning commute. Performance-wise, it takes over two minutes just to boot up and constantly freezes when I try to run more than three Chrome tabs simultaneously. When I tried playing a basic game like Minecraft, the frame rate dropped to a stuttering slideshow while the fans immediately started screaming like a jet engine. The laptop gets dangerously hot on the bottom panel within ten minutes, making it completely impossible to safely rest on my lap without a cooling pad. The battery life is an absolute joke, draining from full to dead in less than two hours of basic word processing during my lectures. To make matters worse, the proprietary charging cable is barely three feet long, forcing me to sit awkwardly glued to the wall outlet. The trackpad is a total nightmare because the cursor randomly jumps across the screen, registering accidental clicks while I am in the middle of typing essays. It completely lacks an HDMI port, meaning I have to buy expensive, annoying adapters just to connect my external monitor for work presentations. The built-in speakers sound completely tinny and muffled, making voices in Zoom meetings completely unintelligible unless I plug in external headphones. The webcam is utterly useless, producing a grainy, purple-tinted video stream that makes me look like a blurry ghost to my coworkers. Technical support was completely unhelpful when I called about these issues, passing me between departments for an hour before abruptly hanging up on me. For such an expensive price tag, the terrible build quality and broken features feel like a total rip-off. It is easily the worst piece of tech I have ever purchased, and I strongly regret not returning it within the store's brief window. Save your hard-earned money and avoid this frustrating, unreliable piece of garbage at all costs."

result = structured_model.invoke(
    f"Provide the sentiment of the following review:\n\n{review}"
)

print(result.sentiment)

chain = msg_prompt | first_model | parser

response = chain.invoke({"customer_review": review , "sentiment": result.sentiment})
print(response)
