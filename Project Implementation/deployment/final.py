# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from Agentic_RAG.Graph import rag_agent
# from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage

# def get_agentic_answer(question,rag_agent):
#     messages = [HumanMessage(question)]

#     result = rag_agent.invoke({"messages": messages})
#     return result['messages'][-1].content



# print(get_agentic_answer("I can not Log in",rag_agent))



import sys
import os
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingface_hub import hf_hub_download
from langchain_core.messages import HumanMessage

# RAG import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Agentic_RAG.Graph import rag_agent


# -------------------------------
# RAG FUNCTION
# -------------------------------
def get_agentic_answer(question):
    messages = [HumanMessage(content=question)]
    result = rag_agent.invoke({"messages": messages})
    return result['messages'][-1].content


# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_ai_components():
    repo_id = "mohamedelkady0/support-ticket-classifier"

    tokenizer = DistilBertTokenizer.from_pretrained(repo_id)
    model = DistilBertForSequenceClassification.from_pretrained(repo_id)

    classes_file_path = hf_hub_download(
        repo_id=repo_id,
        filename="label_classes.npy"
    )
    classes = np.load(classes_file_path, allow_pickle=True)

    return tokenizer, model, classes


# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(page_title="Agent Copilot", page_icon="🧑‍💼", layout="wide")

st.title("🧑‍💼 Support Agent Copilot")
st.caption("AI-powered assistant for ticket classification & resolution support")


# -------------------------------
# SIDEBAR (Agent Tools)
# -------------------------------
st.sidebar.header("⚙️ Agent Tools")
show_debug = st.sidebar.toggle("Show Debug Info", value=True)


# -------------------------------
# LOAD MODELS
# -------------------------------
with st.spinner("Loading AI systems..."):
    tokenizer, model, classes = load_ai_components()


# -------------------------------
# MAIN INPUT (Agent Ticket View)
# -------------------------------
st.subheader("📥 Incoming Ticket")

user_ticket = st.text_area(
    "Paste customer ticket:",
    height=180,
    placeholder="Customer is reporting an issue..."
)

col1, col2 = st.columns([1, 2])

run = col1.button("🔍 Analyze Ticket", type="primary")


# -------------------------------
# MAIN FLOW
# -------------------------------
if run:
    if not user_ticket.strip():
        st.warning("Please enter a ticket.")
        st.stop()

    # ---------------------------
    # CLASSIFICATION
    # ---------------------------
    with st.spinner("Classifying ticket..."):
        model.eval()

        inputs = tokenizer(
            user_ticket,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        top_probs, top_idx = torch.topk(probs, k=3)

        top_issue = classes[top_idx[0][0].item()]
        confidence = top_probs[0][0].item() * 100


    # ---------------------------
    # DASHBOARD LAYOUT
    # ---------------------------
    colA, colB = st.columns([1, 1])


    # ===== LEFT: CLASSIFICATION =====
    with colA:
        st.subheader("🧠 AI Classification")

        st.metric("Predicted Issue", top_issue, f"{confidence:.1f}%")

        st.write("**Top Alternatives**")
        for i in range(1, 3):
            issue = classes[top_idx[0][i].item()]
            conf = top_probs[0][i].item() * 100
            st.write(f"- {issue} ({conf:.1f}%)")

        if show_debug:
            st.bar_chart(probs.numpy()[0])


    # ===== RIGHT: RAG RESPONSE =====
    with colB:
        st.subheader("🤖 Suggested Resolution (RAG)")

        with st.spinner("Generating AI recommendation..."):
            rag_input = f"""
            Ticket: {user_ticket}

            Predicted Category: {top_issue}

            Provide:
            - Root cause analysis
            - Suggested resolution steps
            - Any escalation needed
            """

            rag_answer = get_agentic_answer(rag_input)

        st.write(rag_answer)



    # ---------------------------
    # RAW DATA (DEBUG ONLY)
    # ---------------------------
    if show_debug:
        with st.expander("🔍 Raw Ticket Data"):
            st.json({
                "ticket": user_ticket,
                "prediction": top_issue,
                "confidence": float(confidence)
            })