import asyncio
from typing import Any
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from src.schemas.prd import PRDTemplateSchema
from src.config.settings import llm
from src.utils.supabase.save_prd import save_prd_tx
from src.utils.supabase.client import supabase

UPDATE_SYSTEM_PROMPT = """You are an expert Product Manager updating an existing PRD based on user feedback.

CRITICAL RULES:
1. You will receive the COMPLETE existing PRD data
2. ONLY modify the '{section}' section based on feedback
3. Keep ALL other sections EXACTLY as provided - do not change, omit, or regenerate them
4. Output the FULL PRD with all sections (modified + unchanged)
5. Maintain exact structure: List[Story] for user_stories, List[str] for requirements, etc.

Make changes realistic, specific, and measurable. Add edge cases if relevant.

Available sections: introduction, user_stories, functional_requirements, non_functional_requirements, assumptions, dependencies, risks_and_mitigations, timeline, stakeholders, metrics."""

class UpdatePRDInput(BaseModel):
    """Input schema for updating a PRD section."""
    feedback: str = Field(..., description="Natural language feedback/query for the update (e.g., 'Add offline support')")
    prd_id: str = Field(..., description="Existing PRD ID from Supabase (or thread_id)")
    section: str = Field(..., description="Target section to update (e.g., 'user_stories', 'stakeholders')")

async def update_prd_async(**kwargs: Any) -> str:
    """
    Update a specific section of an existing PRD based on feedback.
    
    Args:
        **kwargs: feedback (str), prd_id (str), section (str).
    
    Returns:
        str: Summary of changes + updated full PRD JSON.
    """
    try:
        input_data = UpdatePRDInput.model_validate(kwargs)
    except ValidationError as e:
        raise ValueError(f"Invalid input: {e}. Need 'feedback', 'prd_id', 'section'.")

    feedback = input_data.feedback
    prd_id = input_data.prd_id
    section = input_data.section

    if not feedback or not prd_id or not section:
        raise ValueError("Feedback, PRD ID, and section are required.")

    # Fetch existing PRD from database
    try:
        existing_data = supabase.table("prds").select("*").eq("id", prd_id).single().execute()
        if not existing_data.data:
            raise ValueError(f"PRD with ID {prd_id} not found in database")
    except Exception as e:
        raise ValueError(f"Failed to fetch PRD {prd_id}: {str(e)}")
    
    existing_prd_dict = existing_data.data
    
    # Convert to Pydantic schema
    try:
        existing_prd = PRDTemplateSchema(**existing_prd_dict)
    except Exception as e:
        raise ValueError(f"Invalid PRD data structure: {str(e)}")
    
    # Calculate old length for diff
    old_section_data = getattr(existing_prd, section, None)
    old_section_len = len(old_section_data) if isinstance(old_section_data, list) else 0
    
    # Structured LLM for section update
    structured_llm = llm.with_structured_output(schema=PRDTemplateSchema)
    
    messages = [
        SystemMessage(content=UPDATE_SYSTEM_PROMPT.format(section=section)),
        HumanMessage(content=f"""EXISTING PRD (keep all sections unchanged except '{section}'):
{existing_prd.model_dump_json(indent=2, exclude_none=True)}

USER FEEDBACK: {feedback}

Output the complete PRD with only '{section}' updated based on feedback. All other sections must remain identical.""")
    ]

    try:
        # Generate updated PRD
        updated_prd = await structured_llm.ainvoke(messages)
        
        # Save with version bump (pass prd_id to update existing record)
        saved_id = await save_prd_tx(updated_prd, prd_id)
        
        # Calculate diff
        new_section_data = getattr(updated_prd, section, None)
        new_section_len = len(new_section_data) if isinstance(new_section_data, list) else 0
        diff = new_section_len - old_section_len
        
        changes_summary = f"Updated '{section}'"
        if diff != 0:
            changes_summary += f" ({diff:+d} items)"
        
        prd_json = updated_prd.model_dump_json(indent=2, exclude_none=True)
        new_version = existing_prd_dict.get('version', 1) + 1
        
        return f"✅ {changes_summary}\n📄 PRD ID: {saved_id}\n🔢 Version: {new_version}\n\nFull updated PRD:\n{prd_json}"
    except Exception as e:
        raise RuntimeError(f"Failed to update PRD: {str(e)}")

# Sync wrapper
def update_prd_sync(**kwargs: Any) -> str:
    return asyncio.run(update_prd_async(**kwargs))

# StructuredTool
update_prd = StructuredTool.from_function(
    func=update_prd_sync,
    name="update_prd",
    description=(
        "Update a specific section of an existing PRD based on feedback. "
        "Parameters: feedback (str, required), prd_id (str, required - use thread_id), section (str, required, e.g., 'stakeholders'). "
        "Returns summary of changes and updated PRD details."
    ),
    args_schema=UpdatePRDInput,
    coroutine=lambda **kwargs: update_prd_async(**kwargs),
)