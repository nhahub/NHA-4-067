from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os






def get_llm():
    load_dotenv()
    api_key = os.getenv('OPEN_ROUTER_API_KEY')
    llm = ChatOpenAI(model="openrouter/free",temperature=0.0,openai_api_base="https://openrouter.ai/api/v1",openai_api_key=api_key)

    return llm