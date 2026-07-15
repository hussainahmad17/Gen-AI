from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
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



vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever =  vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 1}
)

query = "Who is the best bowler among them?"
results = retriever.invoke(query)
for i in results:
    print("Most similar document:")
    print(i.page_content)
    