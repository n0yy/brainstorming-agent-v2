import json
import re
from typing import Any, Optional
from src.utils.request_context import get_thread_id, get_user_id
from src.utils.supabase.client import supabase


SECTION_HEADING_HINTS = {
    "introduction": {"introduction"},
    "user_stories": {"user stories"},
    "functional_requirements": {"functional requirements", "functional requirements core features"},
    "non_functional_requirements": {"non functional requirements", "nonfunctional requirements"},
    "assumptions": {"assumptions"},
    "dependencies": {"dependencies"},
    "risks_and_mitigations": {"risks and mitigations"},
    "timeline": {"timeline", "timeline realistic phases"},
    "stakeholders": {"stakeholders"},
    "metrics": {"metrics"},
}


def _normalize_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _map_section_title(title: str) -> Optional[str]:
    normalized = _normalize_heading(title)
    for field, hints in SECTION_HEADING_HINTS.items():
        for hint in hints:
            if normalized == hint or normalized.startswith(hint):
                return field
    return None


def _split_markdown_sections(content: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(re.finditer(r"^##\s+(.+)$", content, flags=re.MULTILINE))
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        mapped = _map_section_title(title)
        if not mapped:
            continue
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()
        if section_text:
            sections[mapped] = section_text
    return sections


def _parse_prd_markdown(content: str) -> dict[str, str]:
    sections = _split_markdown_sections(content)
    return {key: value.strip() for key, value in sections.items()}


def _serialize_value(value: Any) -> Optional[str]:
    """
    Convert Python objects (list/dict/etc.) into JSON strings so they can be stored in TEXT columns.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


async def save_prd_tx(prd, user_id: str, feature_name: str, prd_id: Optional[str] = None) -> str:
    """
    Save or update PRD to Supabase using the save_prd_tx stored procedure.
    
    Args:
        prd: PRDTemplateSchema instance or raw markdown string
        user_id: Supabase auth/user identifier
        feature_name: Feature identifier/name for the PRD
        prd_id: Optional UUID for update (if None, generates new)
    
    Returns:
        str: UUID of saved PRD
    """
    user_id = user_id or get_user_id()
    prd_id = prd_id or get_thread_id()

    if not user_id:
        raise ValueError("user_id is required to save PRD.")
    if not feature_name or not feature_name.strip():
        raise ValueError("feature_name is required to save PRD.")

    # Extract content from prd
    content: Optional[str] = None
    prd_dict: dict[str, Any] = {}

    if isinstance(prd, str):
        content = prd
    elif hasattr(prd, "model_dump"):
        prd_dict = prd.model_dump(exclude_none=True)  
        content = prd_dict.get("content")
    elif hasattr(prd, "dict"):
        prd_dict = prd.dict(exclude_none=True) 
        content = prd_dict.get("content")
    elif isinstance(prd, dict):
        prd_dict = dict(prd)
        content = prd_dict.get("content")
    else:
        if hasattr(prd, "content"):
            content = getattr(prd, "content", None)
            if isinstance(content, str):
                pass

    if isinstance(content, str) and content.strip():
        parsed_sections = _parse_prd_markdown(content)
        for key, value in parsed_sections.items():
            prd_dict[key] = value  
    
    params = {
        "p_id": prd_id,
        "p_user_id": user_id,
        "p_feature": feature_name.strip(),
        "p_introduction": _serialize_value(prd_dict.get("introduction")),
        "p_user_stories": _serialize_value(prd_dict.get("user_stories")),
        "p_functional_requirements": _serialize_value(prd_dict.get("functional_requirements")),
        "p_non_functional_requirements": _serialize_value(prd_dict.get("non_functional_requirements")),
        "p_assumptions": _serialize_value(prd_dict.get("assumptions")),
        "p_dependencies": _serialize_value(prd_dict.get("dependencies")),
        "p_risks_and_mitigations": _serialize_value(prd_dict.get("risks_and_mitigations")),
        "p_timeline": _serialize_value(prd_dict.get("timeline")),
        "p_stakeholders": _serialize_value(prd_dict.get("stakeholders")),
        "p_metrics": _serialize_value(prd_dict.get("metrics")),
    }
    
    response = supabase.rpc("save_prd_tx", params).execute()
    
    if not response.data:
        raise RuntimeError(f"Failed to save PRD: {response}")
    
    return response.data 