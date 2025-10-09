import json
from typing import AsyncGenerator, Any


def _chunk_to_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content

    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
    elif content:
        parts.append(str(content))

    return "".join(parts)


def _serialize_payload(payload: Any) -> Any:
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    try:
        return json.loads(json.dumps(payload))
    except TypeError:
        return str(payload)


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_response(agent, query: str, config) -> AsyncGenerator[str, None]:
    """
    Stream response dengan format SSE yang terstruktur.
    Frontend bisa membedakan tipe event: tool_start, tool_end, message, error.
    """
    try:
        async for event in agent.astream_events(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            stream_mode="messages",
        ):
            event_name = event["event"]

            if event_name == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                text = _chunk_to_text(chunk)
                if text:
                    yield _sse({"type": "message", "content": text})

            elif event_name == "on_tool_start":
                payload = _serialize_payload(event["data"].get("input"))
                yield _sse(
                    {
                        "type": "tool_start",
                        "tool_name": event.get("name", "unknown"),
                        "args": payload,
                    }
                )

            elif event_name == "on_tool_end":
                payload = _serialize_payload(event["data"].get("output"))
                yield _sse(
                    {
                        "type": "tool_end",
                        "tool_name": event.get("name", "unknown"),
                        "result": payload,
                    }
                )

        yield _sse({"type": "done"})

    except Exception as exc:
        yield _sse({"type": "error", "message": repr(exc)})
