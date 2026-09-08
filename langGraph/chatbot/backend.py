from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Chat node
def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# Checkpointer / Memory
memory_saver = InMemorySaver()


# Build graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


# Compile
workflow = graph.compile(
    checkpointer=memory_saver
)