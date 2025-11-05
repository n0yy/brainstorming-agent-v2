from __future__ import annotations

from typing import Iterable, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.store.base import BaseStore
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.middleware.errors import handle_tool_errors
from src.middleware.model_selector import ModelSelectorMiddleware
from src.tools import execute_bash, generate_prd, get_current_time, http_request, web_search, update_prd 
from src.config.settings import simple_model
from src.middleware.todo import TodoListMiddleware
from code_agent import instructions

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
        execute_bash,
        http_request
    )


def build_agent(
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    store: Optional[BaseStore] = None,
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
    middleware = [
        ModelSelectorMiddleware(),
        handle_tool_errors,
        SummarizationMiddleware(
            model=simple_model,
            max_tokens_before_summary=4000,
            messages_to_keep=20
        ),
        TodoListMiddleware()
    ]

    return create_agent(
        model=simple_model,
        tools=[
            web_search,
            generate_prd,
            update_prd,
            get_current_time,
            execute_bash,
            http_request
        ],
        checkpointer=checkpointer,
        store=store,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        middleware=middleware,
    )


agent = build_agent()
