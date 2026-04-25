from Embedding_model import get_embeder
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
import pickle
from query_improver import improve_user_query



with open('final_documnets.pkl','rb') as f:
    final_documnets = pickle.load(f)

def get_context(query,llm):
    vectorDB = Chroma(
    persist_directory="./chromadb",
    embedding_function=get_embeder()
    )

    retriever = vectorDB.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
    )

    bm25 = BM25Retriever.from_documents(final_documnets)
    bm25.k = 5 

    hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25, retriever],
    weights=[0.4, 0.6]
    )
    final_query = improve_user_query(query,llm)
    context = hybrid_retriever.invoke(final_query)


    return context
