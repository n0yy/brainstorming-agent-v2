from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from psycopg import AsyncConnection
from langgraph.store.postgres.aio import AsyncPostgresStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from src.config.settings import llm, embedding
from src.tools.prd import generate_prd, update_prd 
from src.tools.current_time import get_current_time
from src.utils.stream_response import stream_response
from src.utils.checkpointer import UserAwarePostgresSaver

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
    return {"message": "Simplify Studio v0"}

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
            UserAwarePostgresSaver.from_conn_string(DB_URI)
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

        all_tools = [tavily_search_tool, generate_prd, update_prd, get_current_time]
        all_tools.extend(memory_tools)
        system_prompt = f"""You are an Assistant. Your primary task is to carefully analyze and fully comprehend the user's query. Based on this understanding, identify and select the most appropriate tools from the available set to gather information, perform computations, or retrieve data efficiently. Prioritize tools that directly address the query's needs, such as web searches for factual verification, code execution for calculations or simulations, or X-specific searches for social media insights. Explain your tool selections briefly if relevant, but always proceed to invoke them via function calls when necessary to resolve the query accurately and comprehensively. Ensure all responses are structured, concise, and directly helpful.

CRITICAL - CHECK MEMORY FIRST:
When user asks about themselves:
1. ALWAYS use search_memory tool first
2. NEVER use web search for this
3. If no memory found, tell user honestly

SAVE TO MEMORY:
- User preferences, work context, goals
- Save naturally without asking permission

OTHER TOOLS:
- get_current_time: Always use this to if user asks about (or similar) time 
- TavilySearch: for current events/research (NOT for user info)
- generate_prd/update_prd: when explicitly requested
- PRD operations use prd_id="{thread_id}", user_id="{payload.user_id}"

Always summarize the PRD when PRD is generated.

Respond in the exact language the user is using—detect it from their message and match it perfectly. For example, if they message in Indonesian, reply fully in Indonesian; if in English, stick to English. Be helpful, natural, and super chill—like you're chatting with a close buddy. Use everyday language, light slang (like 'cool', 'totally', 'haha' in English, or 'mantap', 'oke banget', 'wkwk' in Indo), and emojis to keep it fun (but don't overdo it, okay?). Skip formal stuff like 'I suggest'—go for 'I'd say try this' or 'coba deh gini'. Stay accurate and useful, but make the user feel right at home!"""
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
async def get_history(thread_id: str, user_id: str):
    try:
        DB_URI = os.getenv("DB_URI")
        if not DB_URI:
            raise HTTPException(status_code=500, detail="DB_URI not configured")
            
        async with UserAwarePostgresSaver.from_conn_string(DB_URI) as checkpointer:
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            
            async with await AsyncConnection.connect(DB_URI) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT COUNT(*) FROM checkpoints 
                        WHERE thread_id = %s AND user_id = %s
                    """, (thread_id, user_id))
                    result = await cur.fetchone()
                    if result[0] == 0:
                        raise HTTPException(status_code=403, detail="Unauthorized access to thread")
            
            state = await checkpointer.aget(config)
            if state:
                messages = state["channel_values"]["messages"]
                
                formatted_messages = []
                for msg in messages:
                    msg_type = msg.__class__.__name__.replace("Message", "").lower()
                    
                    msg_data = {
                        "type": msg_type,
                        "content": msg.content,
                    }
                    
                    if msg_type == "tool" and hasattr(msg, 'name'):
                        msg_data["tool_name"] = msg.name
                    
                    formatted_messages.append(msg_data)
                
                return formatted_messages

            return []
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
