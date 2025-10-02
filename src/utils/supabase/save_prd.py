from typing import Optional
from src.schemas.prd import PRDTemplateSchema
from src.utils.supabase.client import supabase

async def save_prd_tx(prd: PRDTemplateSchema, prd_id: Optional[str] = None) -> str:
    """
    Save or update PRD to Supabase using the save_prd_tx stored procedure.
    
    Args:
        prd: PRDTemplateSchema instance
        prd_id: Optional UUID for update (if None, generates new)
    
    Returns:
        str: UUID of saved PRD
    """
    # Convert Pydantic model to dict, handling nested models
    prd_dict = prd.model_dump(exclude_none=True)
    
    # Convert lists to JSON-serializable format
    params = {
        "p_id": prd_id,
        "p_feature": prd_dict.get("feature"),
        "p_introduction": prd_dict.get("introduction"),
        "p_user_stories": prd_dict.get("user_stories"),  # Already dict/list from model_dump
        "p_functional_requirements": prd_dict.get("functional_requirements"),
        "p_non_functional_requirements": prd_dict.get("non_functional_requirements"),
        "p_assumptions": prd_dict.get("assumptions"),
        "p_dependencies": prd_dict.get("dependencies"),
        "p_risks_and_mitigations": prd_dict.get("risks_and_mitigations"),
        "p_timeline": prd_dict.get("timeline"),
        "p_stakeholders": prd_dict.get("stakeholders"),
        "p_metrics": prd_dict.get("metrics"),
    }
    
    # Call stored procedure
    response = supabase.rpc("save_prd_tx", params).execute()
    
    if not response.data:
        raise RuntimeError(f"Failed to save PRD: {response}")
    
    return response.data  # Returns UUID