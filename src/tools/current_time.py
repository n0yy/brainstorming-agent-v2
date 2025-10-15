from langchain_core.tools import tool
from datetime import datetime

@tool("current_time")
def get_current_time() -> str:
    """
    Returns the current time in ISO 8601 format.
    """
    return datetime.now().isoformat()