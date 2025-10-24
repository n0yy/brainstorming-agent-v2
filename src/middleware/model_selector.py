from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse
)
from typing_extensions import TypedDict, Literal
from src.config.settings import base_model, simple_model, medium_model, complex_model

class ModelSelector(TypedDict):
    complexity: Literal["simple", "medium", "complex"]

@wrap_model_call
async def async_model_selector(
    request: ModelRequest, 
    handler
) -> ModelResponse:
    
    """
    This middleware selects the best model to use based on the complexity of the query.
    
    It works by using the base model to generate a structured output based on the query.
    The structured output is then used to determine which model to use.
    
    If the structured output is "simple", the simple model is used.
    If the structured output is "medium", the medium model is used.
    If the structured output is "complex", the complex model is used.
    Otherwise, the base model is used.
    
    The selected model is then used to generate the response.
    """
    query = request.state["messages"][-1].content

    model_structured = base_model.with_structured_output(ModelSelector)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a query complexity analyst. "
                "Classify the user query into one of three levels: simple, medium, or complex.\n\n"
                "Rules:\n"
                "- simple: direct questions or tasks solvable in one step (e.g. short factual Q&A, simple code fix).\n"
                "- medium: requires reasoning, multi-step logic, or structured explanation (e.g. function refactor, analysis request).\n"
                "- complex: involves long-form generation, system design, PRD creation, multi-agent orchestration, or high ambiguity.\n\n"
                "Return JSON with a single key 'complexity'."
            ),
        },
        {"role": "user", "content": query},
    ]

    result = model_structured.invoke(messages)

    print(result)

    if result["complexity"] == "simple":
        model = simple_model
    elif result["complexity"] == "medium":
        model = medium_model
    elif result["complexity"] == "complex":
        model = complex_model
    else:
        model = base_model

    request.model = model
    
    return await handler(request)
