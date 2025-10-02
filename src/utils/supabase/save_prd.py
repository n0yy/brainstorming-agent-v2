import inspect
import os
from typing import Optional

from supabase import AsyncClient, acreate_client

from dotenv import load_dotenv

from src.schemas.prd import PRDTemplateSchema

load_dotenv()

_supabase_client: Optional[AsyncClient] = None


async def _get_supabase_client() -> AsyncClient:
    """Initialise Supabase async client once and reuse it."""
    global _supabase_client

    if _supabase_client is None:
        client_candidate = acreate_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_ANON_KEY')
        )

        if inspect.isawaitable(client_candidate):
            client_candidate = await client_candidate

        _supabase_client = client_candidate

    return _supabase_client

async def save_prd_tx(prd: PRDTemplateSchema, prd_id: Optional[str] = None) -> str:
    """
    Save or update PRD dengan atomic transaction via RPC (async).
    """
    data = prd.model_dump(exclude_unset=True)
    p_id = prd_id if prd_id else None
    
    params = {
        'p_id': p_id,
        'p_feature': data['feature'],
        'p_introduction': data['introduction'],
        'p_user_stories': data['user_stories'],
        'p_functional_requirements': data['functional_requirements'],
        'p_non_functional_requirements': data['non_functional_requirements'],
        'p_assumptions': data['assumptions'],
        'p_dependencies': data['dependencies'],
        'p_risks_and_mitigations': data['risks_and_mitigations'],
        'p_timeline': data['timeline'],
        'p_stakeholders': data['stakeholders'],
        'p_metrics': data['metrics']
    }

    try:
        supabase = await _get_supabase_client()
        response = await supabase.rpc('save_prd_tx', params).execute()
        if response.data and len(response.data) > 0:
            saved_id = str(response.data[0])
            print(f"PRD {saved_id} saved/updated with version {data.get('version', 1)}")
            return saved_id
        else:
            raise ValueError(f"RPC failed: {response.get('error', 'No data returned')}")
    except Exception as e:
        raise RuntimeError(f"Transaction error (auto-rollback): {str(e)}")
