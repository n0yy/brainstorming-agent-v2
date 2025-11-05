from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import InjectedToolCallId


class SubTodo(TypedDict):
    """A detailed sub-task with specific implementation instructions."""

    content: str
    """Detailed description of the sub-task including WHAT to implement and HOW to implement it."""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the sub-task."""


class Todo(TypedDict):
    """A main todo item (user story) with content, status, and detailed sub-tasks."""

    content: str
    """The content/description of the main todo item (user story)."""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the main todo item."""

    sub_todos: NotRequired[list[SubTodo]]
    """List of detailed sub-tasks with specific implementation instructions."""


class PlanningState(AgentState):
    """State schema for the todo middleware."""

    todos: NotRequired[list[Todo]]
    """List of todo items for tracking task progress."""


WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list based on user stories from the PRD.

## Structure
Each todo represents a USER STORY with DETAILED sub-tasks that explain WHAT to do and HOW to do it.

## Sub-todo Detail Requirements
Each sub-todo must include:
1. **WHAT**: Clear description of the component/feature to build
2. **HOW**: Specific technical approach and implementation details
3. **WHERE**: Which files/modules to modify or create
4. **ACCEPTANCE**: What defines completion

Example of DETAILED sub-todo:
❌ Bad: "Create API endpoint"
✅ Good: "Create POST /api/products/filter endpoint that accepts price range parameters (min_price, max_price) and returns filtered products. Implementation: Add new route in products.controller.ts, create filterByPrice service method in products.service.ts using Prisma where clause, validate inputs with Zod schema, return paginated response with total count."

## How to Structure Todos

1. **Main Todos (User Stories)**:
   - Format: "As a [user], I want to [action] so that [benefit]"
   - One main todo per user story

2. **Sub-todos (Detailed Implementation Tasks)**:
   - Be EXTREMELY specific about what to build and how
   - Include file names, function names, library choices
   - Mention specific technical patterns or architectures
   - Describe data flow and integration points
   - Include edge cases and validation requirements
   
Example structure:
```
{
  "content": "As a customer, I want to filter products by price so that I can find items within my budget",
  "status": "in_progress",
  "sub_todos": [
    {
      "content": "Create price range slider UI component: Build PriceRangeSlider.tsx using shadcn/ui Slider component with dual thumbs for min/max price selection. Add real-time price labels showing current range. Implement debounced onChange handler (300ms) to avoid excessive API calls. Style with Tailwind including responsive design for mobile. Store selected range in React state and expose onRangeChange callback prop.",
      "status": "completed"
    },
    {
      "content": "Implement backend filter API: Create GET /api/products/filter endpoint in products.controller.ts accepting query params: minPrice, maxPrice, page, limit. Use Prisma query with where: { price: { gte: minPrice, lte: maxPrice } } and include pagination with skip/take. Add input validation using Zod schema (prices must be positive numbers, max > min). Return { products: Product[], total: number, page: number } format. Add index on products.price column for query optimization.",
      "status": "in_progress"
    },
    {
      "content": "Connect filter to product list: In ProductList.tsx, lift price range state up and pass to both PriceRangeSlider and ProductGrid components. Use React Query (useQuery) to fetch filtered products with { queryKey: ['products', { minPrice, maxPrice, page }] }. Implement loading skeleton during fetch. Show 'No products found' state when results empty. Add URL params synchronization using useSearchParams so filter state persists on page refresh.",
      "status": "pending"
    },
    {
      "content": "Add comprehensive testing: Write unit tests for filterByPrice service method covering edge cases (null prices, negative values, inverted range). Create integration test for /api/products/filter endpoint using supertest. Add React Testing Library tests for PriceRangeSlider interaction and ProductList filter integration. Test mobile responsive behavior. Achieve minimum 80% code coverage on new code.",
      "status": "pending"
    }
  ]
}
```

## Task States
- pending: Not started
- in_progress: Currently working on
- completed: Fully finished

Mark first sub-task as in_progress immediately. Main todo completed only when ALL sub-todos completed."""

WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to manage user story implementation from PRDs.

**Critical: Sub-todos must be EXTREMELY DETAILED**

Each sub-todo should read like a mini implementation guide that includes:
- Exact files to create/modify
- Specific functions/components to build
- Libraries or patterns to use
- Technical implementation approach
- Data structures and API contracts
- Validation and error handling needs
- Integration points with other parts
- Testing requirements

**Bad sub-todo example:**
"Create login form"

**Good sub-todo example:**
"Create LoginForm.tsx component using react-hook-form with Zod validation schema. Include email (validated with z.string().email()) and password (min 8 chars) fields. Add loading state during submission. On submit, call POST /api/auth/login with credentials, store JWT token in httpOnly cookie on success. Show error toast using sonner for invalid credentials. Redirect to /dashboard after successful login. Style with Tailwind and use shadcn/ui Input and Button components."

**Structure:**
- Main todo = User story from PRD
- Sub-todos = Detailed implementation steps with technical specifics

**Rules:**
- Extract user stories from PRD
- Write verbose, technical sub-todos (aim for 3-5+ sentences each)
- Include specific file names, function names, library choices
- Describe the "how" not just the "what"
- Update status as you progress through implementation"""


@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Create and manage a structured task list based on user stories with detailed sub-tasks."""
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)],
        }
    )


class TodoListMiddleware(AgentMiddleware):
    """Middleware that provides hierarchical todo list management with detailed sub-tasks.

    This middleware structures todos around user stories from PRDs, with each sub-todo
    containing comprehensive implementation details including technical approach, file names,
    and specific implementation instructions.

    Example:
        ```python
        TodoListMiddleware(
            system_prompt="Generate detailed todos from PRD user stories with technical implementation steps"
        )
        ```

    Args:
        system_prompt: Custom system prompt for todo generation.
        tool_description: Custom description for the write_todos tool.
    """

    state_schema = PlanningState

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
    ) -> None:
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        @tool(description=self.tool_description)
        def write_todos(
            todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> Command:
            """Create and manage a structured task list based on user stories with detailed sub-tasks."""
            return Command(
                update={
                    "todos": todos,
                    "messages": [
                        ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
                    ],
                }
            )

        self.tools = [write_todos]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        request.system_prompt = (
            request.system_prompt + "\n\n" + self.system_prompt
            if request.system_prompt
            else self.system_prompt
        )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        request.system_prompt = (
            request.system_prompt + "\n\n" + self.system_prompt
            if request.system_prompt
            else self.system_prompt
        )
        return await handler(request)