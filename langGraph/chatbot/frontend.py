import os
import uuid

import requests
import streamlit as st

# Use 127.0.0.1, not "localhost": on Windows, "localhost" tries IPv6 (::1) first and
# waits ~2s before falling back to IPv4 on every single request.
API_URL = os.getenv("CHATBOT_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬")
st.title("💬 LangGraph Chatbot")


@st.cache_resource
def get_http() -> requests.Session:
    # One keep-alive session reused across reruns instead of a new TCP connection per call.
    return requests.Session()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_threads() -> list[dict]:
    res = get_http().get(f"{API_URL}/threads", timeout=10)
    res.raise_for_status()
    return res.json()


def fetch_history(thread_id: str) -> list[dict]:
    res = get_http().get(f"{API_URL}/history/{thread_id}", timeout=30)
    res.raise_for_status()
    return [
        {"role": "user" if m["role"] == "human" else "assistant", "content": m["content"]}
        for m in res.json()["messages"]
    ]


# --- State ---
# chats: thread_id -> list of {"role", "content"}. The list for the active thread is the
# same object that gets appended to, so the cache never goes stale when you switch back.
if "chats" not in st.session_state:
    st.session_state["chats"] = {}
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["chats"][st.session_state["thread_id"]] = []


def new_chat():
    thread_id = str(uuid.uuid4())
    st.session_state["thread_id"] = thread_id
    st.session_state["chats"][thread_id] = []


def open_chat(thread_id: str):
    st.session_state["thread_id"] = thread_id
    if thread_id not in st.session_state["chats"]:
        try:
            st.session_state["chats"][thread_id] = fetch_history(thread_id)
        except Exception as e:
            st.session_state["load_error"] = f"Could not load chat: {e}"


# --- Sidebar ---
with st.sidebar:
    st.button("➕ New chat", on_click=new_chat, width="stretch")
    st.caption(f"Thread: `{st.session_state['thread_id']}`")

    st.divider()
    st.subheader("Resume Chat")
    try:
        threads = fetch_threads()
        if threads:
            with st.container(height=400, border=False):
                for t in threads:
                    is_active = t["thread_id"] == st.session_state["thread_id"]
                    st.button(
                        t["title"],
                        key=f"thread_{t['thread_id']}",
                        on_click=open_chat,
                        args=(t["thread_id"],),
                        type="primary" if is_active else "secondary",
                        width="stretch",
                    )
        else:
            st.info("No previous chats found.")
    except requests.exceptions.ConnectionError:
        st.error("Backend is not running (`python backend.py`).")
    except Exception as e:
        st.error(f"Error loading threads: {e}")

if error := st.session_state.pop("load_error", None):
    st.error(error)

# --- Conversation ---
thread_id = st.session_state["thread_id"]
messages = st.session_state["chats"].setdefault(thread_id, [])

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    is_first_message = not messages
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = get_http().post(
                f"{API_URL}/chat_stream",
                json={"message": user_input, "thread_id": thread_id},
                stream=True,
                timeout=(10, 300),
            )
            response.raise_for_status()

            def stream_generator():
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk

            answer_text = st.write_stream(stream_generator())
            messages.append({"role": "assistant", "content": answer_text})

            if is_first_message:
                # A new thread now exists on the backend; refresh the sidebar list.
                fetch_threads.clear()
                st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Please make sure it is running (`python backend.py`).")
        except Exception as e:
            st.error(f"An error occurred: {e}")
