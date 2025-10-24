import asyncio
from typing import Any, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from src.config.settings import medium_model as llm
from src.utils.supabase.save_prd import save_prd_tx
from src.utils.prompts.prd import PRD_SYSTEM_PROMPT 
from src.utils.request_context import get_thread_id, get_user_id
from src.utils.stream_response import _chunk_to_text

from uuid import UUID

class GeneratePRDInput(BaseModel):
    """Input schema for generating a PRD."""
    feature: str = Field(..., description="Feature name or description to generate PRD for")
    user_id: Optional[str] = Field(default=None, description="Supabase user ID for ownership of the PRD")
    prd_id: Optional[str] = Field(default=None, description="Optional existing PRD ID (UUID) when regenerating")

async def generate_prd_async(**kwargs: Any):
    """
    Generate a new PRD based on feature description.
    
    Args:
        **kwargs: feature (str, required), user_id (str, optional, falls back to context), prd_id (str, optional, defaults to thread_id).
    
    Returns:
        str: Generated PRD JSON with ID and version info.
    """
    try:
        input_data = GeneratePRDInput.model_validate(kwargs)
    except ValidationError as e:
        raise ValueError(f"Invalid input: {e}. Need 'feature' parameter.")

    feature = input_data.feature
    user_id = input_data.user_id or get_user_id()
    prd_id = input_data.prd_id or get_thread_id()

    if not feature:
        raise ValueError("Feature description is required.")
    if not user_id:
        raise ValueError("user_id is required.")

    # Validate PRD ID format if provided
    if prd_id:
        try:
            UUID(prd_id)
        except (ValueError, TypeError):
            raise ValueError("prd_id must be a valid UUID string if provided.")

    messages = [
        SystemMessage(content=PRD_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Use the provided schema to create a PRD. "
                "Populate every required property with realistic, specific details. "
                "Feature request: "
                f"{feature}"
            )
        )
    ]

    try:
        full_prd_text = ""
        async for chunk in llm.astream(messages):
            text = _chunk_to_text(chunk)
            if text:
                full_prd_text += text

        save_task = asyncio.create_task(
            save_prd_tx(
                full_prd_text,
                user_id=user_id,
                feature_name=feature,
                prd_id=prd_id,
            )
        )
        await save_task

        summary = (
            "✅ PRD generated successfully!\n"
            f"🔢 Version: 1\n"
            f"🎯 PRD: {full_prd_text}"
        )
        return summary
    except Exception as e:
        raise RuntimeError(f"Failed to generate PRD: {str(e)}")

# Sync wrapper
def generate_prd_sync(**kwargs: Any) -> str:
    return asyncio.run(generate_prd_async(**kwargs))

# StructuredTool
generate_prd = StructuredTool.from_function(
    func=generate_prd_sync,
    name="generate_prd",
    description=(
        "Generate a new Product Requirements Document (PRD) based on feature description. "
        "Parameters: feature (str, required), user_id (str, optional - defaults to context user), prd_id (str, optional existing PRD UUID/default thread_id). "
        "Returns generated PRD with ID for future updates."
    ),
    args_schema=GeneratePRDInput,
    coroutine=lambda **kwargs: generate_prd_async(**kwargs),
)
