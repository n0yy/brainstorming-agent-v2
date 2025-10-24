import asyncio
import json
from typing import Any, Optional, Type
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from src.config.settings import medium_model as llm
from src.utils.supabase.client import supabase
from src.utils.request_context import get_thread_id
from src.utils.stream_response import _chunk_to_text 

UPDATE_SYSTEM_PROMPT = """You are an expert Product Manager updating an existing PRD section based on user feedback.

CRITICAL RULES:
1. You will receive ONLY the existing content for the '{section}' section.
2. Analyze the user feedback carefully and update ONLY the specific parts of the '{section}' that are directly addressed or implied by the feedback. Do NOT make unrelated changes to other parts of the section.
3. Preserve the existing structure, format, and content as much as possible. Only modify, add, or remove elements that are explicitly relevant to the feedback.
4. If the feedback suggests adding new content, integrate it seamlessly into the existing structure without disrupting the overall format.
5. Output the updated section in detailed Markdown format, maintaining the original structure where possible
6. Always use English.
7. Output ONLY the Markdown content for the section. No JSON, no extra text, no headers outside the section."""

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

SECTION_ALIASES: dict[str, str] = {
    "acceptance_criteria": "functional_requirements",
}

def _normalize_section_name(section: str) -> str:
    return section.strip().lower()

def _resolve_section_name(section: str) -> str:
    normalized = _normalize_section_name(section)
    canonical = SECTION_ALIASES.get(normalized, normalized)
    if canonical not in SECTION_FIELD_TYPES:
        available = ", ".join(sorted(SECTION_FIELD_TYPES.keys()))
        raise ValueError(
            f"Unsupported section '{section}'. Available sections: {available}"
        )
    return canonical

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
    return str(value)  # Untuk markdown, langsung str

async def update_prd_async(**kwargs: Any) -> str:
    """
    Update a specific section of an existing PRD based on feedback.
    Collects full LLM output (Markdown) and processes/saves.
    Returns pure LLM Markdown + summary.
    """
    try:
        input_data = UpdatePRDInput.model_validate(kwargs)
    except ValidationError as e:
        raise ValueError(f"Invalid input: {e}. Need 'feedback' and 'section'.")

    feedback = input_data.feedback
    prd_id = input_data.prd_id or get_thread_id()
    requested_section = input_data.section
    section = _resolve_section_name(requested_section)
    display_section = requested_section
    if _normalize_section_name(requested_section) != section:
        display_section = f"{requested_section} (mapped to {section})"

    if not feedback or not section:
        raise ValueError("Feedback and section are required.")
    if not prd_id:
        raise ValueError("PRD ID is required but missing. Make sure to supply thread_id when calling this tool.")

    # Fetch existing section and version
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

    # Prepare messages
    pretty_existing = (
        json.dumps(existing_section_value, ensure_ascii=False, indent=2)
        if isinstance(existing_section_value, (dict, list))
        else str(existing_section_value or "")
    )
    human_content = (
        f"EXISTING SECTION '{section}':\n{pretty_existing}\n\n"
        f"USER FEEDBACK: {feedback}\n\n"
        f"Output the updated section in detailed Markdown format as per the system prompt."
    )
    prompt = UPDATE_SYSTEM_PROMPT.replace("{section}", section)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ]

    # Stream and collect full LLM output
    full_text = ""
    async for chunk in llm.astream(messages):
        text = _chunk_to_text(chunk)
        if text:
            full_text += text

    try:
        updated_section = full_text.strip()
        if not updated_section:
            raise ValueError("LLM output is empty")

        old_len = len(str(existing_section_value)) if existing_section_value else 0
        new_len = len(updated_section)
        changes_summary = f"Updated '{display_section}'"
        if new_len > old_len:
            changes_summary += f" (added details)"

        new_version = existing_version + 1
        serialized_section = _serialize_value(updated_section)
        await asyncio.to_thread(
            lambda: supabase
            .table("prds")
            .update({section: serialized_section, "version": new_version})
            .eq("id", prd_id)
            .execute()
        )

        # Return pure LLM Markdown + summary
        summary = f"\n\n✅ {changes_summary}\n📄 PRD ID: {prd_id}\n🔢 Version: {new_version}"
        return updated_section + summary

    except Exception as e:
        raise RuntimeError(f"Failed to update PRD: {str(e)}")

# Sync wrapper
def update_prd_sync(**kwargs: Any) -> str:
    return asyncio.run(update_prd_async(**kwargs))

update_prd = StructuredTool.from_function(
    func=update_prd_sync,
    name="update_prd",
    description=(
        "Update a specific section of an existing PRD based on feedback. "
        "Generates updated section in detailed Markdown format, saves, and returns Markdown + summary. "
        "Parameters: feedback (str, required), prd_id (str, optional - defaults to thread_id), section (str, required, e.g., 'stakeholders')."
    ),
    args_schema=UpdatePRDInput,
    coroutine=update_prd_async,
)