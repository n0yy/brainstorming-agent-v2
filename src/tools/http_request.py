from langchain_core.tools import tool
from typing import Literal, Any
import httpx
from pydantic import BaseModel, Field


class HttpResponse(BaseModel):
    status: int | None = None
    headers: dict[str, str] | None = None
    data: str | None = None
    error: str | None = None


@tool
async def http_request(
    url: str,
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
    headers: dict[str, str] | None = None,
    data: Any = None,
) -> dict:
    """Make an HTTP request to a URL.
    
    Args:
        url: The URL to make the request to
        method: HTTP method (default: GET)
        headers: HTTP headers
        data: Request body data
    
    Returns:
        Response with status, headers, and data
    """
    try:
        request_headers = {
            "Content-Type": "application/json",
            **(headers or {}),
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                headers=request_headers,
                json=data if data and method != "GET" else None,
                timeout=30.0,
            )
            
            return {
                "status": response.status_code,
                "headers": dict(response.headers),
                "data": response.text,
            }
    
    except Exception as e:
        return {
            "error": str(e),
        }