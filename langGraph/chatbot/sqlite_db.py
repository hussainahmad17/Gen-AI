from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# define the llm

llm = ChatOpenAI(model="gpt-4", temperature=0.7)


# define the state

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# define the node

def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]

    # To enable token-by-token streaming in LangGraph with stream_mode="messages",
    # we use llm.stream() instead of llm.invoke().
    full_response = None
    for chunk in llm.stream(messages):
        if full_response is None:
            full_response = chunk
        else:
            full_response += chunk

    return {"messages": [full_response]}


# define the graph

graph = StateGraph(ChatState)

# add nodes

graph.add_node("chat_node", chat_node)

# add edges

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# checkpointer and db connection

with SqliteSaver.from_conn_string("chatbot.db") as checkpointer:
    # compile the graph
    chatbot = graph.compile(checkpointer=checkpointer)

    # test
    config = {
        "configurable": {
            "thread_id": "thread-1"
        }
    }

    response = chatbot.invoke(
        {"messages": [HumanMessage(content="Hello, how are you?")]}, config=config
    )

    print(response)

