from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    model="azure/gpt-4.1",
    streaming=True, 
    temperature=0.1
)

embedding = OpenAIEmbeddings(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    model="azure/text-embedding-ada-002"
)