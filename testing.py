import asyncio
import uuid

from src.agent import agent
from src.tools.memory import Context
from langgraph.types import Command

user_id = "user-2"
config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
        "user_id": user_id,
    }
}

async def main() -> None:
    context = Context(user_id=user_id)
    
    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, input, "User: "
            )
            if user_input.lower().strip() == 'exit':
                break
            
            inputs = {"messages": [{"role": "user", "content": user_input}]}
            current_tool = None
            has_interrupt = False
            
            # Stream untuk melihat proses
            async for message_chunk, metadata in agent.astream(
                inputs,
                config=config,
                stream_mode="messages",
                context=context,
            ):
                node = metadata.get("langgraph_node")
                if node == "model":
                    if message_chunk.tool_call_chunks:
                        for tool_chunk in message_chunk.tool_call_chunks:
                            name = tool_chunk.get("name")
                            if name:
                                print({"type": "tool_start", "tool_name": name})
                                current_tool = name
                    elif message_chunk.content_blocks:
                        print(message_chunk.content_blocks)    
                elif node == "tools":
                    if message_chunk.content_blocks:
                        print(message_chunk.content_blocks)
                    if current_tool:    
                        print({"type": "tool_end", "tool_name": current_tool})
                        current_tool = None
                
            if has_interrupt:
                print("\n🛑 Graph meminta persetujuan")
                human = input("1. Approve\n2. Reject\n\nUser: ")
                
                command = None
                if human == "1":
                    command = Command(
                        resume={"type": "approve"}
                    )
                elif human == "2":
                    command = Command(
                        resume={
                            "type": "reject",
                            "message": "User menolak permintaan."
                        }
                    )
                
                if command:
                    async for message_chunk, metadata in agent.astream(
                        command,
                        config=config,
                        stream_mode="messages",
                        context=context,
                    ):
                        node = metadata.get("langgraph_node")
                        
                        if node == "model":
                            if message_chunk.content_blocks:
                                print(message_chunk.content_blocks)
                        elif node == "tools":
                            if message_chunk.content_blocks:
                                print(message_chunk.content_blocks)
            
        except KeyboardInterrupt:
            print("\nChat dihentikan.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(main())