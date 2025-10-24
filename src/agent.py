from __future__ import annotations

from typing import Iterable, Optional, Sequence

from langchain.agents import create_agent
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.middleware.errors import handle_tool_errors
from src.middleware.model_selector import async_model_selector
from src.tools.current_time import get_current_time
from src.tools.memory import create_memory
from src.tools.prd import generate_prd, update_prd
from src.tools.web_search import web_search
from src.config.settings import simple_model

DEFAULT_SYSTEM_PROMPT = """You are King Abel, a wise and charismatic AI assistant! 🤖✨

Your personality:
- Friendly and approachable - greet users warmly
- Enthusiastic - show genuine excitement to help
- Clear and informative - explain things simply but thoroughly
- Professional yet fun - balance expertise with charm
- Patient - never rush, always supportive

Communication style:
- Use encouraging language ("Great question!", "I'd be happy to help!")
- Break down complex info into digestible pieces
- Add relevant examples when helpful
- Acknowledge user feelings/context
- End with open invitations for follow-up questions

USER INFORMATION:
- user_id="{user_id}"
- thread_id="{thread_id}"

RESPONSE STYLE:
- Match user's language exactly (Indonesian → Indonesian, English → English)
- Be conversational, friendly, and helpful
- Use appropriate casual expressions naturally
- Stay accurate while being personable
"""


def _base_tools() -> Iterable:
    """List base tools used by the agent."""
    return (
        web_search,
        generate_prd,
        update_prd,
        get_current_time,
        create_memory,
    )


def build_agent(
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
    extra_middleware: Optional[Sequence] = None,
):
    """
    Construct the LangGraph agent with shared middleware and tools.

    Parameters
    ----------
    system_prompt:
        Prompt template used as the system instructions. Accepts `{user_id}` and
        `{thread_id}` placeholders that can be formatted by the caller.
    checkpointer:
        Optional LangGraph checkpointer for persistence.
    store:
        Optional vector store for contextual retrieval.
    """
    middleware = list(extra_middleware or []) + [
        async_model_selector,
        handle_tool_errors,
    ]

    return create_agent(
        model=simple_model,
        tools=list(_base_tools()),
        checkpointer=checkpointer,
        store=store,
        system_prompt=system_prompt,
        middleware=middleware,
    )


# Export default graph instance for CLI tooling. Uses no persistent resources.
agent = build_agent()
