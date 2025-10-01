from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langgraph.utils.config import get_store
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from termcolor import colored
from src.config.settings import llm, embedding
from src.tools.prd import generate_prd

from dotenv import load_dotenv
import os

load_dotenv()

# DEFINE THE TOOLS
tavily_search_tool = TavilySearch(max_results=3)

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = embedding.embed_documents(texts)
    return response

tools = [
    tavily_search_tool,      
    generate_prd,
]

def main():    

    config = {
        "configurable": {
            "thread_id": "1000",
            "user_id": "1",
        }
    }

    DB_URI = os.getenv("DB_URI")
    
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer, \
         PostgresStore.from_conn_string(
             DB_URI,
             index={
                 "dims": 1536,
                 "embed": embed_texts,
             }
         ) as store:
        
        # Setup
        # try:
        #     checkpointer.setup()
        #     store.setup()
        # except Exception as e:
        #     print(colored(f"Error during setup: {e}", "red", attrs=["bold"]))
        #     return

        # Create the memory tool now that store is available
        memory_tool = [
            create_manage_memory_tool(
                namespace=("memories", "{user_id}"),
                instructions=(
                    "Proactively call this tool when "
                    "1. Identify a new USER preference"
                    "2. Receive an explicit USER request"
                    "3. Are working and want to record"
                    "4. Identify that an existing MEMORY"
                    "5. Want to recall a specific MEMORY"
                    "6. Want to forget a specific MEMORY"
                    "7. Want to update a specific MEMORY"
                    "8. the user activity"
                )
            ),
            create_search_memory_tool(namespace=("memories", "{user_id}"))
        ]
        tools.extend(memory_tool)
        
        agent = create_react_agent(
            model=llm,
            tools=tools,
            store=store,
            prompt="You are a helpful assistant. when use generate_prd tool you must summary that PRD.",
            checkpointer=checkpointer
        )

        while True:
            query = input(colored("You: ", "cyan", attrs=["bold"]))
            if query.lower() == "q":
                print(colored("\n👋 Bye!", "red", attrs=["bold"]))
                break

            print(colored("\n" + "─" * 60, "yellow"))
            
            
            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
                config=config,
            ):
                chunk["messages"][-1].pretty_print()

            print(colored("\n" + "─" * 60 + "\n", "yellow"))

if __name__ == "__main__":
    main()