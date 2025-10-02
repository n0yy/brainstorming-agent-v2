import asyncio
from typing import Any
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

import uuid

from src.schemas.prd import PRDTemplateSchema
from src.config.settings import llm
from src.utils.supabase.save_prd import save_prd_tx, _get_supabase_client
from src.utils.prompts.prd import UPDATE_SYSTEM_PROMPT

class UpdatePRDInput(BaseModel):
    """Input schema for updating a PRD section."""
    feedback: str = Field(..., description="Natural language feedback/query for the update (e.g., 'Add offline support')")
    prd_id: str = Field(..., description="Existing PRD ID from Supabase")
    section: str = Field(..., description="Target section to update (e.g., 'user_stories', 'stakeholders')")

async def load_prd_by_id(prd_id: str) -> PRDTemplateSchema:
    """Load existing PRD from Supabase by ID."""
    try:
        # Validate UUID format
        uuid.UUID(prd_id)  # Raise ValueError kalau invalid
    except ValueError:
        raise ValueError(f"Invalid PRD ID format: {prd_id}. Must be a valid UUID (e.g., '123e4567-e89b-12d3-a456-426614174000').")

    supabase = await _get_supabase_client()
    try:
        response = await supabase.table('prds').select('*').eq('id', prd_id).execute()
        if response.data and len(response.data) > 0:
            data = response.data[0]
            return PRDTemplateSchema.model_validate(data)
        else:
            raise ValueError(f"PRD with ID {prd_id} not found.")
    except Exception as e:
        raise RuntimeError(f"Failed to load PRD: {str(e)}")

async def update_prd_async(**kwargs: Any) -> str:
    """
    Update a specific section of an existing PRD based on feedback.
    Accepts **kwargs directly from agent (e.g., feedback=..., prd_id=..., section=...).
    
    Args:
        **kwargs: feedback (str), prd_id (str), section (str).
    
    Returns:
        str: Summary of changes + updated full PRD JSON.
    """
    try:
        # Validate kwargs into Pydantic model (handles unpacked args)
        input_data = UpdatePRDInput.model_validate(kwargs)
    except ValidationError as e:
        raise ValueError(f"Invalid input: {e}. Need 'feedback', 'prd_id', 'section'.")

    feedback = input_data.feedback
    prd_id = input_data.prd_id
    section = input_data.section

    if not feedback or not prd_id or not section:
        raise ValueError("Feedback, PRD ID, and section are required.")

    # Load existing PRD
    existing_prd = await load_prd_by_id(prd_id)
    
    # Get existing section content for context
    existing_section = getattr(existing_prd, section, None)
    if existing_section is None:
        raise ValueError(f"Invalid section: {section}. Available: introduction, user_stories, etc.")

    # Structured LLM for section update
    structured_llm = llm.with_structured_output(schema=PRDTemplateSchema)
    
    messages = [
        SystemMessage(content=UPDATE_SYSTEM_PROMPT.format(section=section)),
        HumanMessage(content=f"Existing {section}: {existing_section}\n\nFeedback: {feedback}\n\nUpdate ONLY {section} and output the full updated PRD with other sections unchanged.")
    ]

    try:
        # Generate updated PRD (LLM copies most, tweaks section)
        updated_prd = await structured_llm.ainvoke(messages)
        
        # Save with version bump
        saved_id = await save_prd_tx(updated_prd, prd_id)
        
        # Simple diff summary
        old_len = len(existing_section) if hasattr(existing_section, '__len__') else 0
        new_len = len(getattr(updated_prd, section, []))
        changes_summary = f"Updated {section} ({new_len - old_len:+} items added/removed) based on feedback."
        
        prd_json = updated_prd.model_dump_json(indent=2, exclude_none=True)
        return f"{changes_summary} PRD updated with ID {saved_id} (version bumped). Full JSON:\n{prd_json}"
    except Exception as e:
        raise RuntimeError(f"Failed to update PRD: {str(e)}")

# Sync wrapper: Accepts **kwargs, runs async
def update_prd_sync(**kwargs: Any) -> str:
    return asyncio.run(update_prd_async(**kwargs))

# StructuredTool: Uses schema for agent parsing, but func handles **kwargs
update_prd = StructuredTool.from_function(
    func=update_prd_sync,  # Sync version with **kwargs
    name="update_prd",
    description=(
        "Update a specific section of an existing PRD based on feedback. "
        "Parameters: feedback (str, required), prd_id (str, required), section (str, required, e.g., 'stakeholders'). "
        "Returns summary of changes and updated PRD details."
    ),
    args_schema=UpdatePRDInput,  # Helps agent parse natural language to named args
    coroutine=lambda **kwargs: update_prd_async(**kwargs),
)