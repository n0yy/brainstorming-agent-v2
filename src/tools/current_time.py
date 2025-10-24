from datetime import datetime
from langchain.tools import tool


@tool("current_time")
def get_current_time() -> str:
    """Get the current date and time in a human-readable format."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
