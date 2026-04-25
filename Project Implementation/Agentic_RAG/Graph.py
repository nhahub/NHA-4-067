import pickle
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import re



with open(r'D:\Intelligent-Support-Ticket\Project Implementation\RAG\final_documnets.pkl' ,'rb') as f:
    documents = pickle.load(f)


def load_vectorstore():
    embedding = OllamaEmbeddings(model='nomic-embed-text')

    vectordb = Chroma(
        persist_directory=r"D:\Intelligent-Support-Ticket\Project Implementation\Agentic_RAG\chromadb_Agentic",
        embedding_function=embedding,
        collection_name="support_tickets"
    )

    return vectordb

vectorDB = load_vectorstore()

retriever = vectorDB.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)


bm25 = BM25Retriever.from_documents(documents)
bm25.k = 5 

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25, retriever],
    weights=[0.4, 0.6]
)


def extract_issue_resolution(text: str):
    issue = ""
    resolution = ""

    # Clean weird wrapping
    text = text.replace("content='", "").replace("'", "").strip()

    # Regex extraction (more reliable)
    issue_match = re.search(r"Issue:\s*(.*?)\s*Resolution:", text, re.DOTALL)
    resolution_match = re.search(r"Resolution:\s*(.*)", text, re.DOTALL)

    if issue_match:
        issue = issue_match.group(1).strip()

    if resolution_match:
        resolution = resolution_match.group(1).strip()

    return issue, resolution


@tool
def retriever_tool(query: str) -> str:
    """
    Retrieves relevant support tickets and returns structured Issue/Resolution.
    """

    docs = hybrid_retriever.invoke(query)

    if not docs:
        return "No relevant support tickets found."

    results = []

    for i, doc in enumerate(docs):
        category = doc.metadata.get("category", "Unknown")
        ticket_id = doc.metadata.get("ticket_id", "N/A")

        issue, resolution = extract_issue_resolution(doc.page_content)

        results.append(
            f"""Ticket {i+1}:
Category: {category}
Ticket ID: {ticket_id}

Issue:
{issue}

Resolution:
{resolution}
"""
        )

    return "\n\n".join(results)

llm = ChatOllama(model="qwen2.5:7b",temperature=0.0) 
tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

system_prompt = """
You are a support assistant that helps customer support agents handle user issues using a ticket knowledge base.

You have access to a retrieval tool that returns relevant support tickets containing Issue and Resolution sections.

Rules:
- Always use the retrieval tool for support-related questions when needed
- Base your answers only on retrieved tickets
- Do not make up information
- You may call the tool multiple times if necessary

When responding:
- Analyze the user’s issue as a support agent would
- Identify relevant Issue(s) from past tickets
- Use the corresponding Resolution(s) to suggest how the agent should handle the case
- Provide a clear, structured, and actionable recommendation for the support agent

Output should be written as guidance to a customer support agent (not directly to the customer).

If no relevant tickets are found, say:
"I couldn't find relevant information in the support tickets."

Do not mention the tool or retrieval process in your final answer.
"""

def call_llm(state: AgentState) -> AgentState:
    """Call LLM with system prompt + messages"""
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node) 

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "tools",
        False: END
    }
)

graph.add_edge("tools", "llm")

rag_agent = graph.compile()



