from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()

base_model = ChatOpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    model="openrouter/qwen/qwen3-8b",
    streaming=False,
    temperature=0.0,
)

simple_model = ChatOpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    model="azure/gpt-oss-20b",
    streaming=True, 
)

medium_model = ChatOpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    model="azure/gpt-oss-120b",
    streaming=True, 
    temperature=0.0
)

complex_model = ChatOpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    model="azure/gpt-oss-120b",
    streaming=True, 
    temperature=0.0
)

embedding = OpenAIEmbeddings(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL"),
    model="azure/text-embedding-ada-002"
)