import sys
import os
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingface_hub import hf_hub_download

# -------------------------------
# PATH SETUP
# -------------------------------
RAG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG"))
sys.path.append(RAG_DIR)
os.chdir(RAG_DIR)

from response import get_response
from LLM import get_llm


# -------------------------------
# RAG FUNCTION
# -------------------------------
def get_agentic_answer(question, llm):
    try:
        return get_response(question, llm)
    except Exception as e:
        return f"⚠️ Could not generate a response: {e}"


# -------------------------------
# LOAD MODELS (cached once)
# -------------------------------
@st.cache_resource
def load_classifier_components():
    repo_id = "mohamedelkady0/support-ticket-classifier"
    tokenizer = DistilBertTokenizer.from_pretrained(repo_id)
    model = DistilBertForSequenceClassification.from_pretrained(repo_id)
    classes_file_path = hf_hub_download(repo_id=repo_id, filename="label_classes.npy")
    classes = np.load(classes_file_path, allow_pickle=True)
    return tokenizer, model, classes


@st.cache_resource
def load_llm():
    return get_llm()


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Agent Copilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# GLOBAL STYLES
# -------------------------------
st.markdown("""
<style>
  /* ── Reset & base ── */
  * { box-sizing: border-box; }

  .stApp {
    background: #0b0d14;
  }

  /* ── Hide default streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Top bar replacement ── */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
    background: #0e1120;
    border-bottom: 0.5px solid #1c2035;
    margin: -1rem -1rem 1.5rem -1rem;
  }
  .topbar-brand { display: flex; align-items: center; gap: 10px; }
  .topbar-icon {
    width: 32px; height: 32px;
    border-radius: 9px;
    background: #4f46e5;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; color: #fff;
  }
  .topbar-name { font-size: 0.92rem; font-weight: 500; color: #e8eaf6; margin: 0; }
  .topbar-sub  { font-size: 0.7rem; color: #5c6080; margin: 0; }
  .status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34,197,94,0.08);
    border: 0.5px solid rgba(34,197,94,0.2);
    color: #4ade80;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem; font-weight: 500;
  }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: #4ade80; }

  /* ── Hero card ── */
  .hero-card {
    background: #111425;
    border: 0.5px solid #1c2137;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.25rem;
  }
  .hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(99,102,241,0.1);
    border: 0.5px solid rgba(99,102,241,0.22);
    color: #818cf8;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
    font-size: 0.68rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 0.45rem;
  }
  .hero-title {
    font-size: 1.3rem; font-weight: 500;
    color: #e8eaf6; margin: 0 0 0.3rem;
  }
  .hero-sub {
    font-size: 0.82rem; color: #4a5070;
    line-height: 1.55; margin: 0; max-width: 500px;
  }
  .stat-cluster { display: flex; gap: 0.75rem; }
  .stat-chip {
    background: #0b0d1a;
    border: 0.5px solid #1c2035;
    border-radius: 12px;
    padding: 0.55rem 1rem;
    text-align: center;
    min-width: 70px;
  }
  .stat-chip .val { font-size: 1.1rem; font-weight: 500; color: #a5b4fc; }
  .stat-chip .lbl { font-size: 0.62rem; color: #3d4260; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

  /* ── Ticket input area ── */
  .ticket-wrap {
    background: #111425;
    border: 0.5px solid #1c2137;
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.25rem;
  }
  .section-label {
    font-size: 0.68rem; font-weight: 500;
    color: #3d4260;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 0.5rem;
  }

  /* ── Main section cards ── */
  .result-card {
    background: #111425;
    border: 0.5px solid #1c2137;
    border-radius: 16px;
    padding: 1.1rem 1.25rem;
    height: 100%;
  }
  .card-head {
    display: flex; align-items: center; gap: 8px;
    padding-bottom: 0.7rem;
    border-bottom: 0.5px solid #191c2e;
    margin-bottom: 0.9rem;
  }
  .card-icon {
    width: 28px; height: 28px; border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
  }
  .icon-purple { background: rgba(99,102,241,0.12); color: #818cf8; }
  .icon-teal   { background: rgba(29,158,117,0.12); color: #34d399; }
  .icon-red    { background: rgba(239,68,68,0.1);   color: #f87171; }
  .card-head-title { font-size: 0.8rem; font-weight: 500; color: #6a7099; }

  /* ── Prediction block ── */
  .pred-issue { font-size: 1rem; font-weight: 500; color: #dde0f8; margin: 0 0 6px; }
  .conf-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 0.17rem 0.6rem;
    border-radius: 999px;
    font-size: 0.71rem; font-weight: 500;
  }
  .conf-high { background: rgba(34,197,94,0.09); color: #4ade80; border: 0.5px solid rgba(34,197,94,0.22); }
  .conf-med  { background: rgba(234,179,8,0.09);  color: #facc15; border: 0.5px solid rgba(234,179,8,0.22); }
  .conf-low  { background: rgba(239,68,68,0.09);  color: #f87171; border: 0.5px solid rgba(239,68,68,0.22); }

  .prog-bg { background: #0d0f1c; border-radius: 999px; height: 5px; overflow: hidden; margin-top: 8px; }
  .prog-fill-high { height: 100%; border-radius: 999px; background: #4ade80; }
  .prog-fill-med  { height: 100%; border-radius: 999px; background: #facc15; }
  .prog-fill-low  { height: 100%; border-radius: 999px; background: #f87171; }

  .alts-header {
    font-size: 0.65rem; color: #2c3050;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin: 0.85rem 0 0.35rem;
  }
  .alt-row {
    display: flex; align-items: center;
    padding: 0.3rem 0;
    border-bottom: 0.5px solid #13162a;
    gap: 8px;
  }
  .alt-row:last-child { border-bottom: none; }
  .alt-name { font-size: 0.78rem; color: #4a5070; flex: 1; }
  .alt-bar-bg { width: 60px; background: #0d0f1c; border-radius: 999px; height: 3px; overflow: hidden; }
  .alt-bar-fill { height: 100%; border-radius: 999px; background: #4f46e5; opacity: 0.45; }
  .alt-pct { font-size: 0.72rem; color: #2c3050; min-width: 34px; text-align: right; }

  /* ── Resolution box ── */
  .res-body {
    border-left: 2px solid #4f46e5;
    padding-left: 0.85rem;
    display: flex; flex-direction: column; gap: 0.75rem;
  }
  .res-block-title {
    font-size: 0.68rem; font-weight: 500;
    color: #818cf8;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
  }
  .res-block-body { font-size: 0.82rem; color: #dde0f8; line-height: 1.65; white-space: pre-wrap; }
  .esc-badge {
    display: inline-flex; align-items: center; gap: 5px;
    margin-top: 0.5rem;
    background: rgba(34,197,94,0.07);
    border: 0.5px solid rgba(34,197,94,0.2);
    color: #4ade80;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.71rem;
  }

  /* ── No-match box ── */
  .no-match-wrap {
    border-left: 2px solid rgba(239,68,68,0.45);
    padding-left: 0.85rem;
  }
  .no-match-title { font-size: 0.88rem; font-weight: 500; color: #f87171; margin-bottom: 0.3rem; }
  .no-match-body  { font-size: 0.82rem; color: #4a3845; line-height: 1.6; }

  /* ── Step stepper ── */
  .step-row { display: flex; gap: 8px; align-items: flex-start; margin-bottom: 5px; }
  .step-num {
    min-width: 18px; height: 18px;
    border-radius: 50%;
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    font-size: 0.62rem; font-weight: 500;
    display: flex; align-items: center; justify-content: center;
    margin-top: 2px; flex-shrink: 0;
  }
  .step-text { font-size: 0.8rem; color: #4a5070; line-height: 1.5; }

  /* ── Streamlit widget overrides ── */
  div.stButton > button {
    background: #4f46e5 !important;
    color: #e0e0ff !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.1rem !important;
    transition: opacity 0.15s ease !important;
  }
  div.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0e1120 !important;
    border-right: 0.5px solid #1c2035 !important;
  }
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] .stMarkdown li {
    color: #4a5070 !important;
    font-size: 0.83rem !important;
    line-height: 1.7 !important;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    color: #6a7099 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
  }
  section[data-testid="stSidebar"] [data-testid="stToggle"] label {
    color: #5c6383 !important;
    font-size: 0.82rem !important;
  }

  /* Text area */
  .stTextArea textarea {
    background: #0b0d14 !important;
    border: 0.5px solid #252840 !important;
    border-radius: 10px !important;
    color: #9ba3c8 !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
  }
  .stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.4) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.08) !important;
  }
  .stTextArea label { color: #3d4260 !important; font-size: 0.75rem !important; }

  /* st.success / warning / info banners */
  .stAlert {
    border-radius: 10px !important;
    border: 0.5px solid !important;
    font-size: 0.83rem !important;
  }

  /* st.subheader / section headings */
  .stMarkdown h2, h2 {
    color: #3d4260 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    margin-bottom: 0.5rem !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: #0e1020 !important;
    border: 0.5px solid #1c2035 !important;
    border-radius: 9px !important;
    color: #4a5070 !important;
    font-size: 0.8rem !important;
  }
  .streamlit-expanderContent {
    background: #0b0d14 !important;
    border: 0.5px solid #1c2035 !important;
    border-radius: 0 0 9px 9px !important;
  }

  /* Bar chart */
  .stBarChart { border-radius: 10px; overflow: hidden; }

  /* Caption */
  .stCaption { color: #2c3050 !important; font-size: 0.72rem !important; }

  /* Divider */
  hr { border-color: #1c2035 !important; }

  /* Spinner */
  .stSpinner > div { color: #818cf8 !important; }
</style>
""", unsafe_allow_html=True)


# -------------------------------
# TOP BAR
# -------------------------------
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-icon">🎯</div>
    <div>
      <p class="topbar-name">Support Agent Copilot</p>
      <p class="topbar-sub">AI-assisted ticket triage</p>
    </div>
  </div>
  <span class="status-pill">
    <span class="status-dot"></span> Systems online
  </span>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.header("Agent Tools")
    show_debug = st.toggle("Show debug info", value=True)
    st.divider()
    st.header("How it works")
    st.markdown(
        "1. RAG searches past resolved tickets\n"
        "2. No match → pipeline stops, no guess\n"
        "3. Match → classify + draft resolution\n"
    )
    st.divider()
    st.caption("Built for triage speed, not agent replacement. Always review before sending.")


# -------------------------------
# LOAD MODELS
# -------------------------------
with st.spinner("Loading AI systems..."):
    tokenizer, model, classes = load_classifier_components()
    llm = load_llm()


# -------------------------------
# HERO CARD
# -------------------------------
st.markdown("""
<div class="hero-card">
  <div>
    <span class="hero-tag">⚡ AI triage</span>
    <p class="hero-title">Analyze incoming tickets</p>
    <p class="hero-sub">Paste a customer ticket — the system retrieves past resolutions first, then classifies only when a confident match is found.</p>
  </div>
  <div class="stat-cluster">
    <div class="stat-chip"><div class="val">91%</div><div class="lbl">Accuracy</div></div>
    <div class="stat-chip"><div class="val">~2s</div><div class="lbl">Avg time</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------
# TICKET INPUT AREA
# -------------------------------
st.markdown('<div class="ticket-wrap">', unsafe_allow_html=True)
st.markdown('<p class="section-label">📥 Incoming Ticket</p>', unsafe_allow_html=True)

user_ticket = st.text_area(
    "ticket_input",
    height=160,
    placeholder='e.g. "Hi, I was charged twice for my subscription this month and I\'d like a refund for the duplicate charge..."',
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 3, 3])
run = col1.button("🔍 Analyze", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------
# MAIN FLOW
# -------------------------------
if run:
    if not user_ticket.strip():
        st.warning("Please enter a ticket before analyzing.")
        st.stop()

    # ── STEP 1: RAG ──
    with st.spinner("Searching past resolutions..."):
        rag_input = f"""
        Ticket: {user_ticket}

        Provide:
        - Root cause analysis
        - Suggested resolution steps
        - Any escalation needed
        """
        rag_answer = get_agentic_answer(rag_input, llm)

    is_no_match = rag_answer.strip().startswith("[NO_MATCH]")
    st.write("")

    if is_no_match:
        # ── NO MATCH ──
        st.warning("No closely matching past ticket was found.")

        colA, colB = st.columns([1, 1.4], gap="medium")

        with colA:
            st.markdown("""
            <div class="result-card">
              <div class="card-head">
                <div class="card-icon icon-red">🧠</div>
                <span class="card-head-title">AI Classification</span>
              </div>
              <div class="no-match-wrap">
                <p class="no-match-title">No match found</p>
                <p class="no-match-body">Classification skipped — no relevant past ticket found to support a reliable prediction.</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            st.markdown(f"""
            <div class="result-card">
              <div class="card-head">
                <div class="card-icon icon-teal">🤖</div>
                <span class="card-head-title">Suggested Resolution</span>
              </div>
              <div class="res-body">
                <div class="res-block-body">{rag_answer}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if show_debug:
            st.write("")
            with st.expander("🔍 Raw ticket data"):
                st.json({"ticket": user_ticket, "prediction": "NO_MATCH", "classification_skipped": True})

    else:
        # ── STEP 2: CLASSIFICATION ──
        with st.spinner("Classifying ticket..."):
            model.eval()
            inputs = tokenizer(user_ticket, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            top_probs, top_idx = torch.topk(probs, k=3)
            top_issue = classes[top_idx[0][0].item()]
            confidence = top_probs[0][0].item() * 100

        if confidence >= 75:
            conf_class, conf_label, prog_class = "conf-high", "High confidence", "prog-fill-high"
        elif confidence >= 45:
            conf_class, conf_label, prog_class = "conf-med",  "Medium confidence", "prog-fill-med"
        else:
            conf_class, conf_label, prog_class = "conf-low",  "Low confidence", "prog-fill-low"

        st.success("Ticket analyzed successfully.")

        colA, colB = st.columns([1, 1.4], gap="medium")

        # ── LEFT: CLASSIFICATION ──
        with colA:
            alt1_issue = classes[top_idx[0][1].item()]
            alt1_conf  = top_probs[0][1].item() * 100
            alt2_issue = classes[top_idx[0][2].item()]
            alt2_conf  = top_probs[0][2].item() * 100

            st.markdown(f"""
            <div class="result-card">
              <div class="card-head">
                <div class="card-icon icon-purple">🧠</div>
                <span class="card-head-title">AI Classification</span>
              </div>
              <p class="pred-issue">{top_issue}</p>
              <span class="conf-pill {conf_class}">{conf_label} · {confidence:.1f}%</span>
              <div class="prog-bg">
                <div class="{prog_class}" style="width:{confidence:.1f}%"></div>
              </div>
              <p class="alts-header">Top alternatives</p>
              <div class="alt-row">
                <span class="alt-name">{alt1_issue}</span>
                <div class="alt-bar-bg"><div class="alt-bar-fill" style="width:{min(alt1_conf*10,100):.0f}%"></div></div>
                <span class="alt-pct">{alt1_conf:.1f}%</span>
              </div>
              <div class="alt-row">
                <span class="alt-name">{alt2_issue}</span>
                <div class="alt-bar-bg"><div class="alt-bar-fill" style="width:{min(alt2_conf*10,100):.0f}%"></div></div>
                <span class="alt-pct">{alt2_conf:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if show_debug:
                st.write("")
                st.caption("Full probability distribution")
                st.bar_chart(probs.numpy()[0])

        # ── RIGHT: RAG RESPONSE ──
        with colB:
            st.markdown(f"""
            <div class="result-card">
              <div class="card-head">
                <div class="card-icon icon-teal">🤖</div>
                <span class="card-head-title">Suggested Resolution</span>
              </div>
              <div class="res-body">
                <div class="res-block-body">{rag_answer}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if show_debug:
            st.write("")
            with st.expander("🔍 Raw ticket data"):
                st.json({"ticket": user_ticket, "prediction": top_issue, "confidence": float(confidence)})

else:
    st.info("Paste a ticket above and click **Analyze** to get started.")