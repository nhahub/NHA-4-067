from LLM import get_llm
from langchain_core.prompts import PromptTemplate

def improve_user_query(input_query, llm):
    prompt_template = PromptTemplate(
        input_variables=["input_query"],
        template="""
You are an expert query rewriting assistant for a support ticket retrieval system.

Your goal is to improve the user query for better search results.

STRICT RULES:
- Preserve the original meaning exactly. Do NOT change intent.
- Fix spelling and grammar mistakes.
- Rewrite the query to be clear, concise, and specific.
- Add relevant keywords ONLY if they are strongly implied.
- Keep important technical terms, error codes, and product names unchanged.
- Do NOT add explanations, only return the improved query.
- Return a single  clear query.

OUTPUT:
<improved_query>

User query:
{input_query}
"""
    )

    fixed_query = llm.invoke(prompt_template.format(input_query=input_query))
    return fixed_query.content.strip()