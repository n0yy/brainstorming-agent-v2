from langchain_core.tools import tool
from langchain_tavily import TavilySearch

import os
from dotenv import load_dotenv
from typing import Any

load_dotenv()

@tool
def web_search(query: str) -> Any:
    """
    This function uses the Firecrawl API to search the web.

    Args:
        query (str): The search query.

    Returns:
        Any: The search results.
    """
    search = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))
    search_results = search.invoke(query)
    return search_results