import asyncio
from typing import Any, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from src.schemas.prd import PRDTemplateSchema
from src.config.settings import llm
from src.utils.supabase.save_prd import save_prd_tx

GENERATE_SYSTEM_PROMPT = """You are an expert Product Manager. Generate a comprehensive Product Requirements Document (PRD) based on the user's feature request.

Requirements:
- Be specific, measurable, and realistic
- Include edge cases and error handling in requirements
- User stories should follow format: "As a [user], I want [goal] so that [benefit]"
- Functional requirements should be clear, testable actions
- Non-functional requirements: performance, security, scalability, etc.
- Timeline should be realistic with milestones
- Identify all stakeholders (users, devs, QA, legal, etc.)
- Metrics should be SMART (Specific, Measurable, Achievable, Relevant, Time-bound)

Output a complete, structured PRD ready for development."""

class GeneratePRDInput(BaseModel):
    """Input schema for generating a PRD."""
    feature: str = Field(..., description="Feature name or description to generate PRD for")
    prd_id: Optional[str] = Field(default=None, description="Optional PRD ID (use thread_id to link to conversation)")

async def generate_prd_async(**kwargs: Any) -> str:
    """
    Generate a new PRD based on feature description.
    
    Args:
        **kwargs: feature (str, required), prd_id (str, optional).
    
    Returns:
        str: Generated PRD JSON with ID and version info.
    """
    try:
        input_data = GeneratePRDInput.model_validate(kwargs)
    except ValidationError as e:
        raise ValueError(f"Invalid input: {e}. Need 'feature' parameter.")

    feature = input_data.feature
    prd_id = input_data.prd_id

    if not feature:
        raise ValueError("Feature description is required.")

    # Structured LLM for PRD generation
    structured_llm = llm.with_structured_output(schema=PRDTemplateSchema)
    
    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(content=f"Feature request: {feature}\n\nGenerate a complete PRD for this feature.")
    ]

    try:
        # Generate new PRD
        new_prd = await structured_llm.ainvoke(messages)
        
        # Save to database (pass prd_id if provided)
        saved_id = await save_prd_tx(new_prd, prd_id)
        
        return f"""✅ PRD generated successfully!
📄 PRD ID: {saved_id}
🔢 Version: 1
🎯 Feature: {feature}

💡 Use this PRD ID ({saved_id}) for future updates with update_prd tool."""
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
        "Parameters: feature (str, required), prd_id (str, optional - use thread_id to link to conversation). "
        "Returns generated PRD with ID for future updates."
    ),
    args_schema=GeneratePRDInput,
    coroutine=lambda **kwargs: generate_prd_async(**kwargs),
)