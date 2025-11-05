from langchain_core.tools import tool
from tavily import TavilyClient

import os
from dotenv import load_dotenv
from typing import Literal


load_dotenv()

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
@tool
def web_search(query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )