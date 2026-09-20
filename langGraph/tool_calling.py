from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
load_dotenv()



llm = ChatOpenAI(model="gpt-4", temperature=0)

# tools

search_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(region="us-en"))

@tool
def calculator(first_number: float, second_number: float, operation: str) -> float:
    """
    A simple calculator tool that performs basic arithmetic operations.

    Args:
        first_number (float): The first number.
        second_number (float): The second number.
        operation (str): The operation to perform. Can be 'add', 'subtract', 'multiply', or 'divide'.

    Returns:
        float: The result of the arithmetic operation.
    """
    if operation == "add":
        return first_number + second_number
    elif operation == "subtract":
        return first_number - second_number
    elif operation == "multiply":
        return first_number * second_number
    elif operation == "divide":
        if second_number == 0:
            raise ValueError("Cannot divide by zero.")
        return first_number / second_number
    else:
        raise ValueError("Invalid operation. Please choose from 'add', 'subtract', 'multiply', or 'divide'.")



tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)


# state

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



# nodes
tool_node = ToolNode(tools)

def chat_node(state: ChatState) -> ChatState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# graph

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# edges

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()


if __name__ == "__main__":
    result = chatbot.invoke({"messages": [HumanMessage(content="What is current political situation in Pakistan? and what is 2 X 3?")]})
    print(result["messages"][-1].content)
