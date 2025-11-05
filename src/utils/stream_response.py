import json
from typing import AsyncGenerator, Any
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, AIMessageChunk

from src.utils.request_context import (
    reset_thread_id,
    reset_user_id,
    set_thread_id,
    set_user_id,
)
from src.tools.memory import Context

def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _extract_text_content(item)
            if text:
                parts.append(text)
        return "".join(parts)
    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "content" in content:
            return _extract_text_content(content["content"])
    if hasattr(content, "text"):
        return _extract_text_content(getattr(content, "text"))
    if hasattr(content, "content"):
        return _extract_text_content(getattr(content, "content"))
    return str(content)


def _chunk_to_text(chunk: Any) -> str:
    """
    Extract text content from a given chunk.

    This function is used to extract text content from a given chunk,
    which could be a string, an AIMessage, an AIMessageChunk, or a dict.
    If the chunk is an AIMessage or an AIMessageChunk, it will extract the text content from it.
    If the chunk is a dict, it will extract the text content from the "content" key.
    If the chunk is None, it will return an empty string.

    :param chunk: The chunk to extract text content from
    :type chunk: Any
    :return: The extracted text content
    :rtype: str
    """
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if hasattr(chunk, "text"):
        text = getattr(chunk, "text")
        if isinstance(text, str):
            return text
    if hasattr(chunk, "content"):
        return _extract_text_content(getattr(chunk, "content"))
    if hasattr(chunk, "message"):
        return _chunk_to_text(getattr(chunk, "message"))
    return str(chunk)


async def stream_response(agent, query: str, config) -> AsyncGenerator[str, None]:
    """
    Stream a response from an LLM agent.

    This function will yield a sequence of SSE messages, each with a type and content.
    The types can be "tool_start", "tool_end", "content", "tool_content", "error", or "done".
    The content will be the relevant data for the given type.

    The function will first yield all the "tool_start" and "content" messages from the model.
    Then, it will yield all the "tool_content" messages from each tool, in the order they were called.
    Finally, it will yield a single "done" message.

    If an exception occurs while streaming, it will yield an "error" message with the exception details.
    """
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    thread_token = set_thread_id(configurable.get("thread_id"))
    user_token = set_user_id(configurable.get("user_id"))
    
    current_tool = None
    
    try:
        async for chunk, _metadata in agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            stream_mode="messages",
            context=Context(user_id=user_token),
        ):
            node = _metadata.get("langgraph_node")  
            if node == "model":
                if chunk.tool_call_chunks:
                    for tool_chunk in chunk.tool_call_chunks:
                        name = tool_chunk.get("name")
                        if name:
                            yield _sse({"type": "tool_start", "tool_name": name})
                            current_tool = name
                elif chunk.content_blocks:
                    for block in chunk.content_blocks:
                        yield _sse({"type": "assistant", "data": block})
            elif node == "tools":
                if chunk.content_blocks:
                    for block in chunk.content_blocks:
                        yield _sse({"type": "tool_content", "data": block})
                if current_tool:
                    yield _sse({"type": "tool_end", "tool_name": current_tool})
                    current_tool = None

        yield _sse({"type": "done"})

    except Exception as exc:
        yield _sse({
            "type": "error",
            "message": str(exc),
            "error_type": type(exc).__name__
        })
    finally:
        reset_thread_id(thread_token)
        reset_user_id(user_token)
