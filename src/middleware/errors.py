from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage


@wrap_tool_call
async def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage( 
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )