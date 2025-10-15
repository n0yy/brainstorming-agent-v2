import asyncio
import json
from typing import Any, Optional, Type
from pydantic import BaseModel, Field, ValidationError, create_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from src.config.settings import llm
from src.utils.supabase.client import supabase
from src.utils.request_context import get_thread_id

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
    prd_id: Optional[str] = Field(default=None, description="Existing PRD ID from Supabase (or thread_id)")
    section: str = Field(..., description="Target section to update (e.g., 'user_stories', 'stakeholders')")

SECTION_FIELD_TYPES: dict[str, Type[Any]] = {
    "user_stories": list[dict[str, Any]],
    "functional_requirements": list[str],
    "non_functional_requirements": list[str],
    "assumptions": list[str],
    "dependencies": list[str],
    "risks_and_mitigations": list[dict[str, Any]],
    "stakeholders": list[str],
    "metrics": list[str],
    "timeline": dict[str, Any],
}


def _infer_field_type(section: str, existing_value: Any) -> Type[Any]:
    mapped = SECTION_FIELD_TYPES.get(section)
    if isinstance(existing_value, str):
        return str
    if isinstance(existing_value, list):
        first_item = existing_value[0] if existing_value else None
        if isinstance(first_item, dict):
            return list[dict[str, Any]]
        if isinstance(first_item, (int, float, bool, str)):
            return list[type(first_item)]
        return list[str]
    if isinstance(existing_value, dict):
        return dict[str, Any]
    if isinstance(existing_value, (int, float, bool)):
        return type(existing_value)
    if mapped is not None:
        return mapped
    return str


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
        raise ValueError(f"Invalid input: {e}. Need 'feedback' and 'section'.")

    feedback = input_data.feedback
    prd_id = input_data.prd_id or get_thread_id()
    section = input_data.section

    if not feedback or not section:
        raise ValueError("Feedback and section are required.")
    if not prd_id:
        raise ValueError("PRD ID is required but missing. Make sure to supply thread_id when calling this tool.")

    select_cols = f"{section},version"
    try:
        existing_data = await asyncio.to_thread(
            lambda: supabase.table("prds").select(select_cols).eq("id", prd_id).limit(1).execute()
        )
    except Exception as e:
        raise ValueError(f"Failed to fetch PRD {prd_id}: {str(e)}")

    if not existing_data.data:
        raise ValueError(f"PRD with ID {prd_id} not found in database")

    existing_row = existing_data.data[0]
    existing_section_value = _deserialize_value(existing_row.get(section))
    existing_version = existing_row.get("version", 0) or 0

    field_type: Type[Any] = _infer_field_type(section, existing_section_value)

    SectionModel = create_model("SectionUpdate", **{section: (field_type, ...)})
    structured_llm = llm.with_structured_output(schema=SectionModel, method="function_calling")

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
        "Parameters: feedback (str, required), prd_id (str, optional - defaults to thread_id), section (str, required, e.g., 'stakeholders'). "
        "Returns summary of changes and updated PRD details."
    ),
    args_schema=UpdatePRDInput,
    coroutine=lambda **kwargs: update_prd_async(**kwargs),
)
