from langgraph.prebuilt import create_react_agent
from psycopg import AsyncConnection
from langgraph.store.postgres.aio import AsyncPostgresStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from src.config.settings import llm, embedding
from src.tools.prd import generate_prd, update_prd 
from src.tools.current_time import get_current_time
from src.utils.stream_response import stream_response
from src.utils.checkpointer import UserAwarePostgresSaver
from src.tools.web_search import web_search

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

import os
from contextlib import AsyncExitStack
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

# ============================================================================
# CONFIGURATION & DEPENDENCIES
# ============================================================================

def get_db_uri() -> str:
    """Dependency untuk mendapatkan DB URI dengan proper error handling"""
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        logger.error("DB_URI not configured in environment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database configuration missing"
        )
    return db_uri


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChatPayload(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="User's chat query")
    user_id: str = Field(..., min_length=1, max_length=100, description="User's ID")

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class PRDInfo(BaseModel):
    prd_id: str
    feature: str
    introduction: Optional[str] = None
    user_stories: Optional[List[str]] = None
    functional_requirements: Optional[List[str]] = None
    non_functional_requirements: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    risks_and_mitigations: Optional[Dict[str, str]] = None
    timeline: Optional[str] = None
    stakeholders: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    version: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MessageResponse(BaseModel):
    type: str
    content: Any
    tool_name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    messages: List[MessageResponse]
    has_prd: bool
    prd: Optional[PRDInfo] = None


class ThreadSummary(BaseModel):
    thread_id: str
    message_count: int
    last_checkpoint_id: str
    has_prd: bool
    prd: Optional[Dict[str, Any]] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embedding function dengan error handling"""
    try:
        return embedding.embed_documents(texts)
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding service error"
        )


def schedule_stack_close(stack: AsyncExitStack) -> None:
    """Cleanup async resources dengan proper error handling"""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(stack.aclose())
    except RuntimeError:
        asyncio.run(stack.aclose())
    except Exception as e:
        logger.error(f"Error closing stack: {str(e)}")


async def get_prd_info(conn: AsyncConnection, thread_id: str, user_id: str) -> Optional[PRDInfo]:
    """Helper untuk fetch PRD info dengan reusable logic"""
    try:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT 
                    id, feature, introduction, user_stories,
                    functional_requirements, non_functional_requirements,
                    assumptions, dependencies, risks_and_mitigations,
                    timeline, stakeholders, metrics, version,
                    created_at, updated_at
                FROM prds 
                WHERE id::text = %s AND user_id = %s
                ORDER BY version DESC
                LIMIT 1
            """, (thread_id, user_id))
            
            row = await cur.fetchone()
            if not row:
                return None
            
            return PRDInfo(
                prd_id=str(row[0]),
                feature=row[1],
                introduction=row[2],
                user_stories=row[3],
                functional_requirements=row[4],
                non_functional_requirements=row[5],
                assumptions=row[6],
                dependencies=row[7],
                risks_and_mitigations=row[8],
                timeline=row[9],
                stakeholders=row[10],
                metrics=row[11],
                version=row[12],
                created_at=row[13],
                updated_at=row[14]
            )
    except Exception as e:
        logger.error(f"Error fetching PRD: {str(e)}")
        return None


# ============================================================================
# ROUTES
# ============================================================================

@router.get("/", include_in_schema=False)
async def root():
    return {"message": "Simplify Studio v0", "status": "healthy"}


@router.post(
    "/chat/{thread_id}",
    response_class=StreamingResponse,
    responses={
        200: {"description": "Streaming chat response"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"}
    }
)
async def chat(
    thread_id: str,
    payload: ChatPayload,
    db_uri: str = Depends(get_db_uri)
):
    """
    Stream chat responses dengan proper resource management.
    
    - **thread_id**: Unique identifier untuk conversation thread
    - **payload**: Chat query dan user_id
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": payload.user_id,
        }
    }

    stack = AsyncExitStack()

    try:
        # Initialize resources dengan async context managers
        checkpointer = await stack.enter_async_context(
            UserAwarePostgresSaver.from_conn_string(db_uri)
        )
        store = await stack.enter_async_context(
            AsyncPostgresStore.from_conn_string(
                db_uri,
                index={"dims": 1536, "embed": embed_texts}
            )
        )

        # Setup tools
        memory_tools = [
            create_manage_memory_tool(namespace=("memories", "{user_id}")),
            create_search_memory_tool(namespace=("memories", "{user_id}"))
        ]

        all_tools = [
            web_search,
            generate_prd,
            update_prd,
            get_current_time,
            *memory_tools
        ]

        system_prompt = f"""You are an AI Assistant designed to understand user queries deeply and select the most appropriate tools to resolve them efficiently.

MEMORY MANAGEMENT (Priority):
- When users ask about themselves: ALWAYS search_memory first, NEVER web search
- Save user preferences, context, and goals naturally without asking
- If no memory found, communicate honestly

TOOL USAGE:
- get_current_time: For time-related queries
- TavilySearch: For current events/research (NOT user information)
- generate_prd/update_prd: When explicitly requested
  * Use prd_id="{thread_id}", user_id="{payload.user_id}"
  * Always summarize PRD after generation

RESPONSE STYLE:
- Match user's language exactly (Indonesian → Indonesian, English → English)
- Be conversational, friendly, and helpful
- Use appropriate casual expressions naturally
- Stay accurate while being personable"""

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
        
        stack = None  # Prevent cleanup in finally block
        return response
        
    except Exception as e:
        logger.error(f"Chat error for thread {thread_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request"
        )
    finally:
        if stack is not None:
            await stack.aclose()


@router.get(
    "/chat/{thread_id}/history",
    response_model=ThreadHistoryResponse,
    responses={
        403: {"description": "Unauthorized access"},
        404: {"description": "Thread not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_history(
    thread_id: str,
    user_id: str,
    db_uri: str = Depends(get_db_uri)
):
    """
    Retrieve chat history dan PRD info untuk thread tertentu.
    
    - **thread_id**: Thread identifier
    - **user_id**: User identifier untuk authorization
    """
    try:
        async with UserAwarePostgresSaver.from_conn_string(db_uri) as checkpointer:
            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            
            # Authorization check & PRD fetch
            async with await AsyncConnection.connect(db_uri) as conn:
                async with conn.cursor() as cur:
                    # Verify user access
                    await cur.execute("""
                        SELECT COUNT(*) FROM checkpoints 
                        WHERE thread_id = %s AND user_id = %s
                    """, (thread_id, user_id))
                    
                    if (await cur.fetchone())[0] == 0:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Unauthorized access to thread"
                        )
                
                # Fetch PRD info
                prd_info = await get_prd_info(conn, thread_id, user_id)
            
            # Get messages from checkpoint
            state = await checkpointer.aget(config)
            
            if not state:
                return ThreadHistoryResponse(
                    thread_id=thread_id,
                    messages=[],
                    has_prd=prd_info is not None,
                    prd=prd_info
                )
            
            messages = state.get("channel_values", {}).get("messages", [])
            
            # Format messages
            formatted_messages = []
            for msg in messages:
                msg_type = msg.__class__.__name__.replace("Message", "").lower()
                
                msg_data = {
                    "type": msg_type,
                    "content": msg.content,
                }
                
                if msg_type == "tool" and hasattr(msg, 'name'):
                    msg_data["tool_name"] = msg.name
                
                if msg_type == "ai" and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_data["tool_calls"] = [
                        {"name": tc.get("name"), "args": tc.get("args")}
                        for tc in msg.tool_calls
                    ]
                
                formatted_messages.append(msg_data)
            
            return ThreadHistoryResponse(
                thread_id=thread_id,
                messages=formatted_messages,
                has_prd=prd_info is not None,
                prd=prd_info
            )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve history"
        )


@router.get(
    "/chat/user/{user_id}/threads",
    response_model=List[ThreadSummary],
    responses={
        500: {"description": "Internal server error"}
    }
)
async def get_user_threads(
    user_id: str,
    db_uri: str = Depends(get_db_uri)
):
    """
    List semua threads milik user, sorted by most recent.
    
    - **user_id**: User identifier
    """
    try:
        async with await AsyncConnection.connect(db_uri) as conn:
            async with conn.cursor() as cur:
                # Optimized query dengan single JOIN
                await cur.execute("""
                    SELECT 
                        c.thread_id,
                        COUNT(DISTINCT c.checkpoint_id) as message_count,
                        MAX(c.checkpoint_id) as last_checkpoint_id,
                        p.id as prd_id,
                        p.feature as prd_feature,
                        p.version as prd_version
                    FROM checkpoints c
                    LEFT JOIN prds p ON p.id::text = c.thread_id 
                        AND p.user_id = c.user_id
                    WHERE c.user_id = %s
                    GROUP BY c.thread_id, p.id, p.feature, p.version
                    ORDER BY MAX(c.checkpoint_id) DESC
                """, (user_id,))
                
                threads = await cur.fetchall()
                
                return [
                    ThreadSummary(
                        thread_id=row[0],
                        message_count=row[1],
                        last_checkpoint_id=row[2],
                        has_prd=row[3] is not None,
                        prd={
                            "prd_id": str(row[3]),
                            "feature": row[4],
                            "version": row[5]
                        } if row[3] else None
                    )
                    for row in threads
                ]
                
    except Exception as e:
        logger.error(f"Error in get_user_threads: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user threads"
        )