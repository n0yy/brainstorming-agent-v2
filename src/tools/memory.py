from langmem import ReflectionExecutor, create_memory_store_manager
from langgraph.store.memory import InMemoryStore

from src.config.settings import llm, embedding

memory_manager = create_memory_store_manager(
    llm,
    namespace=("memories", "episodes", "{user_id}"),
    instructions="Extract examples of successful explanations, capturing the full chain of reasoning. Be concise in your explanations and precise in the logic of your reasoning.",
    enable_inserts=True,
    enable_deletes=True
)

executor = ReflectionExecutor(memory_manager)

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = embedding.embed_documents(texts)
    return response

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": embed_texts,
    }
)