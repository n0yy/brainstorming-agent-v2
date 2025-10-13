import asyncio
import json
from typing import Any, Optional, Type
from pydantic import BaseModel, Field, ValidationError, create_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from src.config.settings import llm
from src.utils.supabase.client import supabase

UPDATE_SYSTEM_PROMPT = """You are an expert Product Manager updating an existing PRD section based on user feedback.

CRITICAL RULES:
1. You will receive ONLY the existing content for the '{section}' section
2. Update ONLY the '{section}' section based on feedback
3. Return STRICTLY a JSON object with a single key '{section}' and the updated value of the correct type
4. Do NOT include any other sections in the output
5. Maintain exact structure: for example, List[Story] for user_stories, List[str] for requirements, plain string for timeline, etc.

Make changes realistic, specific, and measurable. Add edge cases if relevant."""

class UpdatePRDInput(BaseModel):
    """Input schema for updating a PRD section."""
    feedback: str = Field(..., description="Natural language feedback/query for the update (e.g., 'Add offline support')")
    prd_id: str = Field(..., description="Existing PRD ID from Supabase (or thread_id)")
    section: str = Field(..., description="Target section to update (e.g., 'user_stories', 'stakeholders')")

LIST_SECTION_HINTS = {
    "user_stories",
    "functional_requirements",
    "non_functional_requirements",
    "assumptions",
    "dependencies",
    "risks_and_mitigations",
    "timeline",
    "stakeholders",
    "metrics",
}


def _deserialize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def _serialize_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


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

    try:
        select_cols = f"{section},version"
        existing_data = await asyncio.to_thread(
            lambda: supabase.table("prds").select(select_cols).eq("id", prd_id).single().execute()
        )
        if not existing_data.data:
            raise ValueError(f"PRD with ID {prd_id} not found in database")
    except Exception as e:
        raise ValueError(f"Failed to fetch PRD {prd_id}: {str(e)}")

    existing_section_value = _deserialize_value(existing_data.data.get(section))
    existing_version = existing_data.data.get("version", 0) or 0

    # Determine field type for the requested section
    if isinstance(existing_section_value, list):
        field_type: Type[Any] = list
    elif isinstance(existing_section_value, dict):
        field_type = dict
    elif existing_section_value is None and section in LIST_SECTION_HINTS:
        field_type = list
    else:
        field_type = str

    # Build dynamic Pydantic model for structured output { section: field_type }
    SectionModel = create_model("SectionUpdate", **{section: (field_type, ...)})
    structured_llm = llm.with_structured_output(schema=SectionModel)

    # Prepare concise messages focusing only on the target section
    pretty_existing = (
        json.dumps(existing_section_value, ensure_ascii=False) if isinstance(existing_section_value, (dict, list)) else str(existing_section_value or "")
    )
    human_content = (
        f"EXISTING SECTION '{section}':\n{pretty_existing}\n\n"
        f"USER FEEDBACK: {feedback}\n\n"
        f"Return strictly JSON with key '{section}' containing the updated value."
    )
    messages = [
        SystemMessage(content=UPDATE_SYSTEM_PROMPT.format(section=section)),
        HumanMessage(content=human_content),
    ]

    try:
        # Generate updated section only
        section_result = await structured_llm.ainvoke(messages)
        updated_section = section_result.model_dump().get(section)

        # Calculate simple diff summary for list-type sections
        old_len = len(existing_section_value) if isinstance(existing_section_value, list) else None
        new_len = len(updated_section) if isinstance(updated_section, list) else None
        changes_summary = f"Updated '{section}'"
        if old_len is not None and new_len is not None:
            diff = new_len - old_len
            if diff != 0:
                changes_summary += f" ({diff:+d} items)"

        # Persist only the updated section and bump version (non-blocking)
        new_version = existing_version + 1
        serialized_section = _serialize_value(updated_section)
        await asyncio.to_thread(
            lambda: supabase
            .table("prds")
            .update({section: serialized_section, "version": new_version})
            .eq("id", prd_id)
            .execute()
        )

        return f"✅ {changes_summary}\n📄 PRD ID: {prd_id}\n🔢 Version: {new_version}"
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
