from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


class SimpleContextCompressionRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever

    def _compress_document(self, query: str, document: Document) -> Document:
        query_terms = {word.lower() for word in query.split() if len(word) > 3}
        sentences = [sentence.strip() for sentence in document.page_content.split(".") if sentence.strip()]

        if not query_terms:
            return document

        matching_sentences = [
            sentence for sentence in sentences
            if any(term in sentence.lower() for term in query_terms)
        ]

        if not matching_sentences:
            return Document(page_content="", metadata=document.metadata)

        return Document(
            page_content=". ".join(matching_sentences),
            metadata=document.metadata,
        )

    def invoke(self, query: str):
        documents = self.base_retriever.invoke(query)
        compressed_documents = [self._compress_document(query, document) for document in documents]
        return [document for document in compressed_documents if document.page_content]



docs = [
    Document(
        page_content=(
            "Python is a popular programming language used for web development, "
            "automation, data science, and artificial intelligence. It emphasizes "
            "readability and has a large ecosystem of libraries."
        ),
        metadata={
            "id": 1,
            "topic": "python",
            "relevance": "irrelevant",
        },
    ),

    Document(
        page_content=(
            "The Amazon rainforest contains millions of plant and animal species. "
            "It plays an important role in regional rainfall patterns and global "
            "carbon storage."
        ),
        metadata={
            "id": 2,
            "topic": "rainforest",
            "relevance": "irrelevant",
        },
    ),

    Document(
        page_content=(
            "A traditional pizza is prepared using dough, tomato sauce, cheese, "
            "and toppings. It is baked at a high temperature until the crust becomes "
            "crisp and the cheese melts."
        ),
        metadata={
            "id": 3,
            "topic": "cooking",
            "relevance": "irrelevant",
        },
    ),

    Document(
        page_content=(
            "The solar system consists of the Sun, eight planets, dwarf planets, "
            "moons, asteroids, and comets. Earth is the third planet from the Sun."
        ),
        metadata={
            "id": 4,
            "topic": "astronomy",
            "relevance": "irrelevant",
        },
    ),

    Document(
        page_content=(
            "Football is played by two teams that attempt to score by moving a ball "
            "into the opposing team's goal. Rules and field dimensions vary between "
            "different forms of the sport."
        ),
        metadata={
            "id": 5,
            "topic": "sports",
            "relevance": "irrelevant",
        },
    ),
]


vector_store = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    collection_name="context-compression-demo",
)

base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
context_compression_retriever = SimpleContextCompressionRetriever(
    base_retriever=base_retriever,
)

query = "What is the solar system?"
results = context_compression_retriever.invoke(query)
for i in results:
    print("Most similar document:")
    print(i.page_content)