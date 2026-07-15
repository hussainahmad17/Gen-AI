from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()


docs = [
    Document(
       page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and exceptional run-scoring ability, he has been a key player for Royal Challengers Bangalore."
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm leadership and elegant batting style."
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicket-keeping, and tactical captaincy are legendary."
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over bowling."
    ),
]

