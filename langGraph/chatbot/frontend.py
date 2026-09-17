import uuid
import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Backend API configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬")
st.title("💬 LangGraph Chatbot")

# State management
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Local cache for restored threads to avoid redundant API calls
if "history_cache" not in st.session_state:
    st.session_state["history_cache"] = {}

if "resume_thread_selection" not in st.session_state:
    st.session_state["resume_thread_selection"] = "None"

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

with st.sidebar:
    st.caption(f"Thread: `{st.session_state['thread_id']}`")
    if st.button("New chat"):
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.session_state["resume_thread_selection"] = "None"
        st.rerun()

    st.divider()
    st.subheader("Resume Chat")
    try:
        threads_res = requests.get(f"{API_URL}/threads")
        threads_res.raise_for_status()
        threads = threads_res.json()

        if threads:
            # Map descriptive titles to thread_ids for the selectbox
            thread_options = {f"{t['title']} ({t['thread_id'][:8]})": t['thread_id'] for t in threads}

            # Use a key and a value linked to session_state to avoid the 'auto-reload' bug
            selected_label = st.selectbox(
                "Select a previous chat:",
                options=["None"] + list(thread_options.keys()),
                key="resume_chat_selector"
            )

            # Update the state variable based on the selector
            if selected_label != st.session_state.get("resume_thread_selection"):
                st.session_state["resume_thread_selection"] = selected_label

                if selected_label != "None":
                    selected_thread = thread_options[selected_label]
                    if selected_thread != st.session_state["thread_id"]:
                        # Check cache first
                        if selected_thread in st.session_state["history_cache"]:
                            history = st.session_state["history_cache"][selected_thread]
                        else:
                            # Load history from API
                            hist_res = requests.get(f"{API_URL}/history/{selected_thread}")
                            hist_res.raise_for_status()
                            history = hist_res.json()["messages"]
                            # Store in cache
                            st.session_state["history_cache"][selected_thread] = history

                        # Rebuild messages from role-based history
                        new_messages = []
                        for msg in history:
                            if msg["role"] == "human":
                                new_messages.append(HumanMessage(content=msg["content"]))
                            else:
                                new_messages.append(AIMessage(content=msg["content"]))

                        st.session_state["thread_id"] = selected_thread
                        st.session_state["messages"] = new_messages
                        st.rerun()
        else:
            st.info("No previous chats found.")
    except Exception as e:
        st.error(f"Error loading threads: {e}")

# render the conversation so far
for message in st.session_state["messages"]:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# handle a new turn
user_input = st.chat_input("Ask me anything...")

if user_input:
    # 1. Display user message
    user_message = HumanMessage(content=user_input)
    st.session_state["messages"].append(user_message)
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Call Backend API (Streaming)
    with st.chat_message("assistant"):
        try:
            payload = {
                "message": user_input,
                "thread_id": st.session_state["thread_id"]
            }
            # Use stream=True to handle the streaming response
            response = requests.post(f"{API_URL}/chat_stream", json=payload, stream=True)
            response.raise_for_status()

            # use st.write_stream to render the response as it comes in
            def stream_generator():
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk

            answer_text = st.write_stream(stream_generator())
            st.session_state["messages"].append(AIMessage(content=answer_text))

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend server. Please make sure it is running (`python backend.py`).")
        except Exception as e:
            st.error(f"An error occurred: {e}")
