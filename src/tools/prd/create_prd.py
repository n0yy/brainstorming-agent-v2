import asyncio
import uuid
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool

from src.schemas.prd import PRDTemplateSchema
from src.config.settings import llm
from src.utils.supabase.save_prd import save_prd_tx
from src.utils.prompts.prd import PRD_SYSTEM_PROMPT


async def generate_prd_async(query: str) -> str:
    """
    Generate a PRD from a query, save/update to Supabase, and return details.
    
    Args:
        query: Natural language query for PRD.
        prd_id: Optional existing ID for update.
    
    Returns:
        str: Saved PRD ID and JSON dump.
    """
    prd_id = str(uuid.uuid4())

    structured_llm = llm.with_structured_output(schema=PRDTemplateSchema)
    messages = [
        SystemMessage(content=PRD_SYSTEM_PROMPT),
        HumanMessage(content=f"Generate a comprehensive PRD for: {query}")
    ]

    try:
        prd = await structured_llm.ainvoke(messages)
        await save_prd_tx(prd, prd_id)
        return f"PRD {'updated' if prd_id else 'saved'} with ID {prd_id}."
    except Exception as e:
        raise RuntimeError(f"Failed to generate PRD: {str(e)}")

# Tool: Required 'func' for sync + 'coroutine' for async (LangGraph compatible)
generate_prd = Tool(
    name="generate_prd",
    description=(
        "Generate a comprehensive Product Requirements Document (PRD) from a user query, "
        "persist it to Supabase, and return the saved PRD details. Summarize the PRD in your response."
    ),
    func=lambda query: asyncio.run(generate_prd_async(query)),  
    coroutine=generate_prd_async,
)