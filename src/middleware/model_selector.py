from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from src.config.settings import base_model, simple_model, medium_model, complex_model


class ModelComplexity(BaseModel):
    complexity: Literal["simple", "medium", "complex"] = Field(
        description="Complexity level of the query"
    )


class ModelSelectorMiddleware(AgentMiddleware):
    """
    Selects appropriate model based on query complexity.
    Uses base_model to analyze complexity, then routes to:
    - simple_model: one-step tasks
    - medium_model: multi-step reasoning
    - complex_model: long-form generation, system design
    """

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        messages = request.state.get("messages", [])
        if not messages:
            return handler(request)

        query = messages[-1].content

        try:
            classifier = base_model.with_structured_output(
                ModelComplexity,
            ).with_config(
                {
                    "callbacks": [],
                    "metadata": {"internal_run": "model_selector"},
                }
            )
            
            classification_prompt = [
                {
                    "role": "system",
                    "content": (
                        "Classify query complexity into: simple, medium, or complex.\n\n"
                        "Rules:\n"
                        "- simple: direct questions, single-step tasks (e.g. 'What is X?', 'Fix this typo')\n"
                        "- medium: multi-step logic, analysis, refactoring (e.g. 'Compare A and B', 'Optimize this function')\n"
                        "- complex: long-form content, system design, PRD, high ambiguity (e.g. 'Design a distributed system', 'Write a comprehensive guide')\n"
                    ),
                },
                {"role": "user", "content": query},
            ]

            result = classifier.invoke(classification_prompt)
            
            if result.complexity == "simple":
                selected_model = simple_model
            elif result.complexity == "medium":
                selected_model = medium_model
            elif result.complexity == "complex":
                selected_model = complex_model
            else:
                selected_model = base_model
            
            request.model = selected_model
            return handler(request)

        except Exception:
            return handler(request)


    async def awrap_model_call(
        self, request: ModelRequest, handler
    ) -> ModelResponse:
        messages = request.state.get("messages", [])
        if not messages:
            return await handler(request)

        query = messages[-1].content

        try:
            classifier = base_model.with_structured_output(
                ModelComplexity,
                include_raw=False
            ).with_config(
                {
                    "callbacks": [],
                    "metadata": {"internal_run": "model_selector"},
                }
            )
            
            classification_prompt = [
                {
                    "role": "system",
                    "content": (
                        "Classify query complexity into: simple, medium, or complex.\n\n"
                        "Rules:\n"
                        "- simple: direct questions, single-step tasks (e.g. 'What is X?', 'Fix this typo')\n"
                        "- medium: multi-step logic, analysis, refactoring (e.g. 'Compare A and B', 'Optimize this function')\n"
                        "- complex: long-form content, system design, PRD, high ambiguity (e.g. 'Design a distributed system', 'Write a comprehensive guide')\n"
                    ),
                },
                {"role": "user", "content": query},
            ]

            result = await classifier.ainvoke(classification_prompt)
            
            if result.complexity == "simple":
                selected_model = simple_model
            elif result.complexity == "medium":
                selected_model = medium_model
            elif result.complexity == "complex":
                selected_model = complex_model
            else:
                selected_model = base_model
            
            request.model = selected_model
            return await handler(request)

        except Exception:
            return await handler(request)
