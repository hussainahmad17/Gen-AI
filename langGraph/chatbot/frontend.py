import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from backend import chatbot

st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬")
st.title("💬 LangGraph Chatbot")

# one thread per browser session -> the checkpointer keeps that thread's history

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

with st.sidebar:
    st.caption(f"Thread: `{st.session_state['thread_id']}`")
    if st.button("New chat"):
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.rerun()

# render the conversation so far

for message in st.session_state["messages"]:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# handle a new turn

user_input = st.chat_input("Ask me anything...")

if user_input:
    user_message = HumanMessage(content=user_input)
    st.session_state["messages"].append(user_message)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        def stream_answer():
            for chunk, _metadata in chatbot.stream(
                {"messages": [user_message]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if chunk.content:
                    yield chunk.content

        answer = st.write_stream(stream_answer())

    st.session_state["messages"].append(AIMessage(content=answer))
