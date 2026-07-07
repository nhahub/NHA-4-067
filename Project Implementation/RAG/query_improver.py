from LLM import get_llm
from langchain_core.prompts import PromptTemplate


def improve_user_query(input_query, llm):
    prompt_template = PromptTemplate(
        input_variables=["input_query"],
        template="""You are an expert query rewriting assistant for a customer support ticket retrieval system (RAG).

Your goal is to rewrite the raw customer query into a single, clear, search-optimized query that will be used to retrieve relevant past support tickets via semantic + keyword search.

STRICT RULES:
- Preserve the original meaning and intent exactly. Never add information, assumptions, or details that weren't said or strongly implied.
- Fix spelling and grammar mistakes.
- If the message contains multiple issues (e.g. a delivery problem AND a billing problem), do NOT drop either one - combine them into a single concise query that still captures both topics, rather than picking only one.
- Keep technical terms, error codes, order/ticket IDs, and product names EXACTLY as written. Do not reformat, translate, or "correct" them.
- Remove filler, greetings, and pleasantries ("hii", "can u check", "I was wondering if") - keep only the substantive content.
- Do NOT answer the question. Do NOT explain your changes. Return ONLY the rewritten query, nothing else.
- The result must be a single line of text with no quotation marks, labels, or formatting around it.

EXAMPLES:

Customer query: "hii my oder #12345 never arrved and also i think yall charged me twice can u check"
Improved query: Order #12345 not delivered and customer charged twice

Customer query: "how do i reset my passwrd"
Improved query: How to reset password

Customer query: "getting error code E-4021 when i try to checkout"
Improved query: Error code E-4021 during checkout

Now rewrite this customer query:

{input_query}

Improved query:"""
    )

    fixed_query = llm.invoke(prompt_template.format(input_query=input_query))
    return fixed_query.content.strip()