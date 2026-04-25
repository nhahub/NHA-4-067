import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingface_hub import hf_hub_download
# 1. Set up the visual page configuration
st.set_page_config(page_title="Support Ticket AI", page_icon="🎫", layout="centered")

# 2. CACHE THE MODEL
@st.cache_resource
def load_ai_components():
    repo_id = "mohamedelkady0/support-ticket-classifier"
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(repo_id)
    model = DistilBertForSequenceClassification.from_pretrained(repo_id)
    
    # Load classes file
    classes_file_path = hf_hub_download(
        repo_id=repo_id,
        filename="label_classes.npy"
    )
    classes = np.load(classes_file_path, allow_pickle=True)
    
    return tokenizer, model, classes


# 3. UI
st.title("🎫 Support Ticket Classifier")
st.write("Powered by a fine-tuned DistilBERT Transformer.")

# Load model with spinner
with st.spinner("Loading AI Model into memory... (This takes a moment on startup)"):
    tokenizer, model, classes = load_ai_components()

# Input
user_ticket = st.text_area(
    "Paste the customer's message below:",
    height=150,
    placeholder="E.g., I'm getting a 504 error when I try to export my billing history."
)

# Button
if st.button("Classify Issue", type="primary"):
    if not user_ticket.strip():
        st.warning("Please enter a ticket description first!")
    else:
        with st.spinner("Analyzing semantics..."):
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

            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)

            top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)

            # Results
            st.subheader("Analysis Results:")

            # Top prediction
            top_issue = classes[top_indices[0][0].item()]
            top_conf = top_probs[0][0].item() * 100

            st.success(f"**Primary Match:** {top_issue} ({top_conf:.1f}% confidence)")

            # Alternatives
            st.write("**Alternative Possibilities:**")
            for i in range(1, 3):
                issue = classes[top_indices[0][i].item()]
                conf = top_probs[0][i].item() * 100
                st.write(f"- {issue}: {conf:.1f}%")