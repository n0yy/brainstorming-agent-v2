import json
import re
from typing import Any, Optional
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
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        section_text = content[start:end].strip()
        if section_text:
            sections[mapped] = section_text
    return sections


def _parse_bullet_list(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        bullet_match = re.match(r"[-*+]\s+(.*)", stripped)
        if bullet_match:
            items.append(bullet_match.group(1).strip())
    if not items:
        collapsed = text.strip()
        if collapsed:
            items.append(collapsed)
    return items


def _parse_numbered_list(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        num_match = re.match(r"\d+[.)]\s+(.*)", stripped)
        if num_match:
            items.append(num_match.group(1).strip())
    if not items:
        collapsed = text.strip()
        if collapsed:
            items.append(collapsed)
    return items


def _extract_subsection(block: str, heading: str) -> str:
    pattern = rf"####\s+{re.escape(heading)}\s*\n"
    match = re.search(pattern, block)
    if not match:
        return ""
    start = match.end()
    following = re.search(r"\n(?:####|###)\s+", block[start:])
    end = start + following.start() if following else len(block)
    return block[start:end].strip()


def _parse_user_stories(section: str) -> list[dict[str, Any]]:
    stories: list[dict[str, Any]] = []
    pattern = re.compile(r"###\s+User Story\s+(\d+)(.*?)(?=\n###\s+User Story|\Z)", flags=re.DOTALL)
    for match in pattern.finditer(section):
        story_number = int(match.group(1))
        block = match.group(2).strip()
        description = _extract_subsection(block, "Description")
        actors_block = _extract_subsection(block, "Actors / Persona")
        pre_condition_block = _extract_subsection(block, "Pre-Condition")
        flow_block = _extract_subsection(block, "Done When (Flow)")
        exception_block = _extract_subsection(block, "Exception Handling")
        acceptance_block = _extract_subsection(block, "Acceptance Criteria")
        dod_block = _extract_subsection(block, "Definition of Done")

        story = {
            "id": story_number,
            "title": f"User Story {story_number}",
            "description": description,
            "actors": _parse_bullet_list(actors_block),
            "pre_conditions": _parse_bullet_list(pre_condition_block),
            "flow": _parse_numbered_list(flow_block),
            "exception_handling": _parse_bullet_list(exception_block),
            "acceptance_criteria": _parse_bullet_list(acceptance_block),
            "definition_of_done": _parse_bullet_list(dod_block),
        }
        stories.append(story)
    if not stories:
        collapsed = section.strip()
        if collapsed:
            stories.append({"raw": collapsed})
    return stories


def _normalize_dict_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _parse_markdown_table(section: str) -> list[dict[str, str]]:
    table_lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    if len(table_lines) < 2:
        return []

    header_parts = [part.strip() for part in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in table_lines[1:]:
        stripped = line.strip()
        if set(stripped.replace("|", "").replace(" ", "")) <= {"-"}:
            continue
        values = [part.strip() for part in stripped.strip("|").split("|")]
        if len(values) != len(header_parts):
            continue
        row = { _normalize_dict_key(header_parts[idx]): values[idx] for idx in range(len(header_parts)) }
        rows.append(row)
    return rows


def _parse_timeline_section(section: str) -> Any:
    phases = _parse_markdown_table(section)
    summary: Optional[str] = None
    for line in section.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("|"):
            if stripped.lower().startswith("total"):
                _, _, remainder = stripped.partition(":")
                summary = remainder.strip() or stripped
                break
    if phases or summary:
        data: dict[str, Any] = {}
        if phases:
            data["phases"] = phases
        if summary:
            data["summary"] = summary
        return data
    return section.strip()


def _parse_prd_markdown(content: str) -> dict[str, Any]:
    sections = _split_markdown_sections(content)
    parsed: dict[str, Any] = {}

    introduction = sections.get("introduction")
    if introduction:
        parsed["introduction"] = introduction.strip()

    user_stories_text = sections.get("user_stories")
    if user_stories_text:
        parsed["user_stories"] = _parse_user_stories(user_stories_text)

    func_req_text = sections.get("functional_requirements")
    if func_req_text:
        parsed["functional_requirements"] = _parse_bullet_list(func_req_text)

    non_func_text = sections.get("non_functional_requirements")
    if non_func_text:
        parsed["non_functional_requirements"] = _parse_bullet_list(non_func_text)

    assumptions_text = sections.get("assumptions")
    if assumptions_text:
        parsed["assumptions"] = _parse_bullet_list(assumptions_text)

    dependencies_text = sections.get("dependencies")
    if dependencies_text:
        parsed["dependencies"] = _parse_bullet_list(dependencies_text)

    risks_text = sections.get("risks_and_mitigations")
    if risks_text:
        parsed["risks_and_mitigations"] = _parse_markdown_table(risks_text)

    timeline_text = sections.get("timeline")
    if timeline_text:
        parsed["timeline"] = _parse_timeline_section(timeline_text)

    stakeholders_text = sections.get("stakeholders")
    if stakeholders_text:
        parsed["stakeholders"] = _parse_bullet_list(stakeholders_text)

    metrics_text = sections.get("metrics")
    if metrics_text:
        parsed["metrics"] = _parse_bullet_list(metrics_text)

    return parsed


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
        prd: PRDTemplateSchema instance
        user_id: Supabase auth/user identifier
        feature_name: Feature identifier/name for the PRD
        prd_id: Optional UUID for update (if None, generates new)
    
    Returns:
        str: UUID of saved PRD
    """
    if not user_id:
        raise ValueError("user_id is required to save PRD.")
    if not feature_name or not feature_name.strip():
        raise ValueError("feature_name is required to save PRD.")

    # Attempt to extract dictionary data from the PRD object
    prd_dict: dict[str, Any]
    content: Optional[str] = None

    if hasattr(prd, "model_dump"):
        prd_dict = prd.model_dump(exclude_none=True)  # type: ignore[attr-defined]
    elif hasattr(prd, "dict"):
        prd_dict = prd.dict(exclude_none=True)  # type: ignore[call-arg]
    elif isinstance(prd, dict):
        prd_dict = dict(prd)
    else:
        prd_dict = {}

    if isinstance(prd, str):
        content = prd
    elif content is None:
        content = prd_dict.get("content") if isinstance(prd_dict, dict) else None
        if not content and hasattr(prd, "content"):
            raw_content = getattr(prd, "content", None)
            if isinstance(raw_content, str):
                content = raw_content

    if isinstance(content, str) and content.strip():
        parsed_sections = _parse_prd_markdown(content)
        for key, value in parsed_sections.items():
            existing = prd_dict.get(key)
            if existing is None or (isinstance(existing, str) and not existing.strip()) or (isinstance(existing, (list, dict)) and not existing):
                prd_dict[key] = value
        prd_dict.setdefault("content", content)
    
    # Convert lists to JSON-serializable format
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
    
    # Call stored procedure
    response = supabase.rpc("save_prd_tx", params).execute()
    
    if not response.data:
        raise RuntimeError(f"Failed to save PRD: {response}")
    
    return response.data  # Returns UUID
