from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


class UserAwarePostgresSaver(AsyncPostgresSaver):
    """
    Custom checkpointer yang otomatis inject user_id ke tabel checkpoints
    """
    
    async def aput(
        self,
        config: dict,
        checkpoint: dict,
        metadata: dict,
        new_versions: dict,
    ) -> dict:
        result = await super().aput(config, checkpoint, metadata, new_versions)
        
        user_id = config.get("configurable", {}).get("user_id")
        thread_id = config.get("configurable", {}).get("thread_id")
        
        if user_id and thread_id and self.conn:
            try:
                async with self.conn.cursor() as cur:
                    await cur.execute(
                        """
                        UPDATE checkpoints 
                        SET user_id = %s 
                        WHERE thread_id = %s 
                        AND checkpoint_id = (
                            SELECT checkpoint_id 
                            FROM checkpoints 
                            WHERE thread_id = %s 
                            ORDER BY checkpoint_id DESC 
                            LIMIT 1
                        )
                        """,
                        (user_id, thread_id, thread_id)
                    )
                await self.conn.commit()
            except Exception as e:
                print(f"Warning: Failed to update user_id in checkpoint: {e}")
        
        return result