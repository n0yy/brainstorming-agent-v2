from langgraph.checkpoint.postgres import PostgresSaver
from langchain.tools import tool, ToolRuntime

from dataclasses import dataclass
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DB_URI")

@dataclass
class Context:
    user_id: str

class EpisodicMemory(BaseModel):
    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(
        ...,
        description="Internal reasoning process and observations of the agent in the episode that let it arrive"
        ' at the correct action and result. "I ..."',
    )
    action: str = Field(
        ...,
        description="What was done, how, and in what format. (Include whatever is salient to the success of the action). I ..",
    )
    result: str = Field(
        ...,
        description="Outcome and retrospective. What did you do well? What could you do better next time? I ...",
    )

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    @tool("create_memory")
    async def create_memory(episode: EpisodicMemory, runtime: ToolRuntime[Context]) -> str:
        """
        Store an episode of memory in the database.

        Parameters:
            episode (EpisodicMemory): The episode to store.
            runtime (ToolRuntime[Context]): The runtime context.

        Returns:
            str: A success message indicating that the memory was created.
        """
        store = runtime.store
        user_id = runtime.context.user_id

        await store.aput(("episode",), user_id, episode)
        return "Memory created"