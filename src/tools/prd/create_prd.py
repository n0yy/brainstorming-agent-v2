import asyncio
from typing import Any, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from src.config.settings import llm
from src.utils.supabase.save_prd import save_prd_tx
from src.utils.prompts.prd import PRD_SYSTEM_PROMPT 

from uuid import UUID

class GeneratePRDInput(BaseModel):
    """Input schema for generating a PRD."""
    feature: str = Field(..., description="Feature name or description to generate PRD for")
    user_id: str = Field(..., description="Supabase user ID for ownership of the PRD")
    prd_id: Optional[str] = Field(default=None, description="Optional existing PRD ID (UUID) when regenerating")

async def generate_prd_async(**kwargs: Any) -> str:
    """
    Generate a new PRD based on feature description.
    
    Args:
        **kwargs: feature (str, required), user_id (str, required), prd_id (str, optional).
    
    Returns:
        str: Generated PRD JSON with ID and version info.
    """
    try:
        input_data = GeneratePRDInput.model_validate(kwargs)
    except ValidationError as e:
        raise ValueError(f"Invalid input: {e}. Need 'feature' parameter.")

    feature = input_data.feature
    user_id = input_data.user_id
    prd_id = input_data.prd_id

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
        # Generate new PRD
        new_prd = await llm.ainvoke(messages)

        # Save to database (pass prd_id if provided)
        await save_prd_tx(new_prd, user_id=user_id, feature_name=feature, prd_id=prd_id)
        
        return f"""✅ PRD generated successfully!
🔢 Version: 1
🎯 Feature: {feature}
💡 PRD: {new_prd}"""
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
        "Parameters: feature (str, required), user_id (str, required), prd_id (str, optional existing PRD UUID). "
        "Returns generated PRD with ID for future updates."
    ),
    args_schema=GeneratePRDInput,
    coroutine=lambda **kwargs: generate_prd_async(**kwargs),
)
