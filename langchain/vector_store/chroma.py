from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Create LangChain documents for IPL players

doc1 = Document(
     page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and exceptional run-scoring ability, he has been a key player for Royal Challengers Bangalore.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm leadership and elegant batting style.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicket-keeping, and tactical captaincy are legendary.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over bowling.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him invaluable.",
    metadata={"team": "Chennai Super Kings"}
)

documents = [doc1, doc2, doc3, doc4, doc5]




vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="mychromaDB",
    collection_name="my_sample_collection"
)

vector_store.add_documents(documents)

# now we will make similarity search

results = vector_store.similarity_search(
    query="Who is the best bowler among them?",
    k=1
)

print("Most similar document:")
for result in results:
    print(result.page_content)
    print("Metadata:", result.metadata)