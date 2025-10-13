from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from src.config.settings import llm, embedding
from src.tools.prd import generate_prd, update_prd 
from src.utils.stream_response import stream_response

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os
from contextlib import AsyncExitStack
import asyncio

load_dotenv()

router = APIRouter()

# DEFINE THE TOOLS
tavily_search_tool = TavilySearch(max_results=3)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = embedding.embed_documents(texts)
    return response


def schedule_stack_close(stack: AsyncExitStack) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(stack.aclose())
    else:
        loop.create_task(stack.aclose())

class ChatPayload(BaseModel):
    query: str = Field(..., min_length=1, description="The user's chat query")
    user_id: str = Field(..., min_length=1, description="The user's ID")

# API ROUTES
@router.get("/")
async def root():
    return {"message": "PM Assistant API"}

@router.post("/api/chat/{thread_id}")
async def chat(thread_id: str, payload: ChatPayload):
    """
    Endpoint dengan proper resource management menggunakan AsyncExitStack.
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": payload.user_id,
        }
    }

    DB_URI = os.getenv("DB_URI")
    if not DB_URI:
        raise HTTPException(status_code=500, detail="DB_URI not configured")

    stack = AsyncExitStack()

    try:
        # Enter async contexts
        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(DB_URI)
        )
        store = await stack.enter_async_context(
            AsyncPostgresStore.from_conn_string(
                DB_URI,
                index={"dims": 1536, "embed": embed_texts}
            )
        )

        # Create tools
        memory_tools = [
            create_manage_memory_tool(
                namespace=("memories", "{user_id}"),
            ),
            create_search_memory_tool(namespace=("memories", "{user_id}"))
        ]

        all_tools = [tavily_search_tool, generate_prd, update_prd]
        all_tools.extend(memory_tools)
        # Create agent with Simpli persona
        system_prompt = f"""You are Simpli, PM assistant. Thread: {thread_id}

CRITICAL - CHECK MEMORY FIRST:
When user asks about themselves:
1. ALWAYS use search_memory tool first
2. NEVER use web search for this
3. If no memory found, tell user honestly

SAVE TO MEMORY:
- User preferences, work context, goals
- Save naturally without asking permission

OTHER TOOLS:
- TavilySearch: for current events/research (NOT for user info)
- generate_prd/update_prd: when explicitly requested
- PRD operations use prd_id="{thread_id}", user_id="{payload.user_id}"

Respond in user's language. Be helpful and natural."""
        
        agent = create_react_agent(
            model=llm,
            tools=all_tools,
            checkpointer=checkpointer,
            store=store,
            prompt=system_prompt
        )

        response = StreamingResponse(
            stream_response(agent, payload.query, config),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
            background=BackgroundTask(schedule_stack_close, stack)
        )
        
        stack = None
        return response
        
    finally:
        if stack is not None:
            await stack.aclose()

@router.get("/api/chat/{thread_id}/history")
async def get_history(thread_id: str):
    try:
        DB_URI = os.getenv("DB_URI")
        if not DB_URI:
            raise HTTPException(status_code=500, detail="DB_URI not configured")
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            config = {"configurable": {"thread_id": thread_id}}
            state = await checkpointer.aget(config)
        
            if state:
                return {"messages": state.get("messages", [])}
            return {"messages": []}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
