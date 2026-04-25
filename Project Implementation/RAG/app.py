import streamlit as st
import time
from response import get_response
from LLM import get_llm

# --------------------------
# Load LLM once
# --------------------------
llm = get_llm()

# --------------------------
# UI Setup
# --------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")


# --------------------------
# Input
# --------------------------
user_input = st.text_input("Ask me something:")

if st.button("Send") and user_input:

    # Create placeholder for bot response
    placeholder = st.empty()

    # --------------------------
    # 1. Spinner while waiting
    # --------------------------
    with st.spinner("Thinking... 🤔"):
        try:
            response = get_response(user_input, llm)
        except Exception as e:
            response = f"[Error] Could not generate response: {e}"

    # --------------------------
    # 2. Typing animation
    # --------------------------
    displayed_text = ""
    for char in response:
        displayed_text += char
        placeholder.markdown(f"**Bot:** {displayed_text}")
        time.sleep(0.01)
