from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from DB import get_context
from query_improver import improve_user_query

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_response(query,llm):
    prompt_template = """
        You are an expert technical support assistant.

        Answer the user's question using ONLY the provided context.

        STRICT RULES:
        - Use ONLY the context. Do NOT hallucinate.
        - If the answer is not clearly supported, respond EXACTLY:
        [I can not answer now !]
        - Be polite, clear, and actionable.
        - Keep the answer concise (1–2 sentences or short steps).
        - Use simple Markdown ONLY if helpful:
        - bullet points (-)
        - numbered steps (1. 2. 3.)
        - Do NOT over-format.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = prompt | llm | StrOutputParser()
    final_query = improve_user_query(query,llm)
    context = get_context(query,llm)

    response = chain.invoke({
    "context":format_docs(context),
    'question' : final_query
    })
    return response

    