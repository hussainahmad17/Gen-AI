from typing import Annotated, TypedDict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import uvicorn
import sqlite3
from contextlib import asynccontextmanager

load_dotenv()

# --- LangGraph Logic ---

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    # We use invoke here because LangGraph's stream_mode="messages"
    # will handle the token-by-token streaming from the LLM.
    response = llm.invoke(messages)
    return {"messages": [response]}

# Create the graph
workflow = StateGraph(ChatState)
workflow.add_node("chat_node", chat_node)
workflow.add_edge(START, "chat_node")
workflow.add_edge("chat_node", END)

# Global variable to hold the compiled chatbot and checkpointer
chatbot = None
checkpointer_ctx = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot, checkpointer_ctx
    # Initialize SqliteSaver using the context manager
    checkpointer_ctx = SqliteSaver.from_conn_string("chatbot.db")
    checkpointer = checkpointer_ctx.__enter__()

    # Create metadata table for chat titles
    conn = sqlite3.connect("chatbot.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_metadata (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
    """)
    conn.commit()
    conn.close()

    # Compile the graph with the checkpointer
    chatbot = workflow.compile(checkpointer=checkpointer)

    yield

    # Clean up
    checkpointer_ctx.__exit__(None, None, None)

app = FastAPI(lifespan=lifespan)

# --- API Models ---

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str

class HistoryMessage(BaseModel):
    role: str
    content: str

class HistoryResponse(BaseModel):
    messages: List[HistoryMessage]

class ThreadResponse(BaseModel):
    thread_id: str
    title: str

# --- Endpoints ---

@app.get("/threads", response_model=List[ThreadResponse])
async def list_threads():
    try:
        conn = sqlite3.connect("chatbot.db")
        cursor = conn.cursor()
        # Use the metadata table for faster listing and meaningful titles
        cursor.execute("SELECT thread_id, title FROM chat_metadata")
        threads = [{"thread_id": row[0], "title": row[1]} for row in cursor.fetchall()]
        conn.close()
        return threads
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching threads: {str(e)}")

@app.get("/history/{thread_id}", response_model=HistoryResponse)
async def get_history(thread_id: str):
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = chatbot.get_state(config)
        messages = state.values.get("messages", [])

        history = []
        for m in messages:
            role = "ai" if m.type == "ai" else "human"
            history.append(HistoryMessage(role=role, content=m.content))

        return HistoryResponse(messages=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    config = {"configurable": {"thread_id": request.thread_id}}

    async def event_generator():
        try:
            # Use stream_mode="messages" to stream tokens from the LLM
            # We pass the input as a dict with the messages key
            for msg, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                stream_mode="messages"
            ):
                # We only want to stream the content of the messages
                if msg.content:
                    yield msg.content
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        # invoke the chatbot
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )

        # --- Auto-titling Logic ---
        # Check if a title already exists for this thread
        conn = sqlite3.connect("chatbot.db")
        cursor = conn.cursor()
        cursor.execute("SELECT title FROM chat_metadata WHERE thread_id = ?", (request.thread_id,))
        row = cursor.fetchone()

        if row is None:
            # This is the first message in the thread. Generate a title.
            # We ask the LLM to summarize the user's first message into a short title.
            title_prompt = f"Summarize the following user message into a short, meaningful chat title (max 5 words): {request.message}"
            title_response = llm.invoke(title_prompt)
            title = title_response.content.strip('"')

            cursor.execute("INSERT INTO chat_metadata (thread_id, title) VALUES (?, ?)", (request.thread_id, title))
            conn.commit()

        conn.close()
        # ---------------------------

        # Get the last message content
        last_message = result["messages"][-1]
        return ChatResponse(response=last_message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
