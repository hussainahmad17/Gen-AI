import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, List, TypedDict

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

load_dotenv()

DB_PATH = os.getenv("CHATBOT_DB", "chatbot.db")

# --- LangGraph Logic ---

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenAI(model=os.getenv("CHAT_MODEL", "gpt-4"), temperature=0.7)
# Titles are a side job, so use a small, fast model for them.
title_llm = ChatOpenAI(model=os.getenv("TITLE_MODEL", "gpt-4o-mini"), temperature=0)

def chat_node(state: ChatState) -> ChatState:
    # stream_mode="messages" picks up the tokens from this call via callbacks.
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(ChatState)
workflow.add_node("chat_node", chat_node)
workflow.add_edge(START, "chat_node")
workflow.add_edge("chat_node", END)

chatbot = None

# --- Chat metadata (titles) ---
# One shared connection guarded by a lock instead of opening a new one per request.
meta_conn: sqlite3.Connection | None = None
meta_lock = threading.Lock()

def init_metadata(conn: sqlite3.Connection):
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_metadata (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
    """)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(chat_metadata)")}
    if "updated_at" not in columns:
        conn.execute("ALTER TABLE chat_metadata ADD COLUMN updated_at REAL DEFAULT 0")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_metadata_updated ON chat_metadata(updated_at DESC)")
    conn.commit()

def make_title(text: str, max_len: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"

def backfill_missing_titles():
    """Register threads that exist in the checkpointer but have no metadata row."""
    with meta_lock:
        missing = [row[0] for row in meta_conn.execute("""
            SELECT DISTINCT thread_id FROM checkpoints
            WHERE thread_id NOT IN (SELECT thread_id FROM chat_metadata)
        """)]
    for thread_id in missing:
        messages = chatbot.get_state({"configurable": {"thread_id": thread_id}}).values.get("messages", [])
        first_human = next((m.content for m in messages if m.type == "human"), "")
        with meta_lock:
            meta_conn.execute(
                "INSERT OR IGNORE INTO chat_metadata (thread_id, title, updated_at) VALUES (?, ?, ?)",
                (thread_id, make_title(first_human) or "Untitled chat", 0),
            )
            meta_conn.commit()

def touch_thread(thread_id: str, first_message: str) -> bool:
    """Insert or bump a thread. Returns True if the thread is new."""
    with meta_lock:
        cur = meta_conn.execute(
            "UPDATE chat_metadata SET updated_at = ? WHERE thread_id = ?", (time.time(), thread_id)
        )
        is_new = cur.rowcount == 0
        if is_new:
            # Instant placeholder title so the chat shows up in the list right away.
            meta_conn.execute(
                "INSERT INTO chat_metadata (thread_id, title, updated_at) VALUES (?, ?, ?)",
                (thread_id, make_title(first_message), time.time()),
            )
        meta_conn.commit()
    return is_new

def generate_title(thread_id: str, message: str):
    """Runs after the response is sent, so it never delays the user."""
    try:
        prompt = f"Summarize the following user message into a short, meaningful chat title (max 5 words). Reply with the title only: {message}"
        title = title_llm.invoke(prompt).content.strip().strip('"')
        if title:
            with meta_lock:
                meta_conn.execute("UPDATE chat_metadata SET title = ? WHERE thread_id = ?", (title, thread_id))
                meta_conn.commit()
    except Exception:
        pass  # keep the placeholder title

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot, meta_conn
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        checkpointer.setup()
        checkpointer.conn.execute("PRAGMA journal_mode=WAL")
        checkpointer.conn.execute("PRAGMA synchronous=NORMAL")

        meta_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        init_metadata(meta_conn)

        chatbot = workflow.compile(checkpointer=checkpointer)
        backfill_missing_titles()

        yield

        meta_conn.close()

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
# Endpoints that do blocking work (SQLite, sync LLM calls) are plain `def`, so FastAPI
# runs them in a threadpool instead of freezing the event loop for every other request.

@app.get("/threads", response_model=List[ThreadResponse])
def list_threads():
    try:
        with meta_lock:
            rows = meta_conn.execute(
                "SELECT thread_id, title FROM chat_metadata ORDER BY updated_at DESC"
            ).fetchall()
        return [{"thread_id": r[0], "title": r[1] or "Untitled chat"} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching threads: {str(e)}")

@app.get("/history/{thread_id}", response_model=HistoryResponse)
def get_history(thread_id: str):
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    try:
        state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        history = [
            HistoryMessage(role="ai" if m.type == "ai" else "human", content=m.content)
            for m in messages
        ]
        return HistoryResponse(messages=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.post("/chat_stream")
def chat_stream(request: ChatRequest, background_tasks: BackgroundTasks):
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    config = {"configurable": {"thread_id": request.thread_id}}
    if touch_thread(request.thread_id, request.message):
        background_tasks.add_task(generate_title, request.thread_id, request.message)

    # A sync generator: StreamingResponse iterates it in a threadpool, so the blocking
    # LangGraph stream doesn't stall other requests (e.g. loading the chat list).
    def event_generator():
        try:
            for msg, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=request.message)]},
                config=config,
                stream_mode="messages",
            ):
                if msg.content:
                    yield msg.content
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain", background=background_tasks)

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")

    config = {"configurable": {"thread_id": request.thread_id}}
    try:
        result = chatbot.invoke({"messages": [HumanMessage(content=request.message)]}, config=config)
        if touch_thread(request.thread_id, request.message):
            background_tasks.add_task(generate_title, request.thread_id, request.message)
        return ChatResponse(response=result["messages"][-1].content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
