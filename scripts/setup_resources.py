import asyncio
import os

from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import AsyncConnection

from src.config.settings import embedding
from typing import Any


PRD_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS prds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    feature TEXT NOT NULL,
    introduction TEXT,
    user_stories TEXT,
    functional_requirements TEXT,
    non_functional_requirements TEXT,
    assumptions TEXT,
    dependencies TEXT,
    risks_and_mitigations TEXT,
    timeline TEXT,
    stakeholders TEXT,
    metrics TEXT,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prds_user_id ON prds (user_id);
CREATE INDEX IF NOT EXISTS idx_prds_user_created ON prds (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_prds_user_stories ON prds USING GIN (user_stories gin_trgm_ops);

DROP INDEX IF EXISTS idx_prds_user_feature;
DROP INDEX IF EXISTS idx_prds_feature;

CREATE INDEX IF NOT EXISTS idx_prds_user_feature_hash ON prds (user_id, md5(feature));
CREATE INDEX IF NOT EXISTS idx_prds_feature_trgm ON prds USING GIN (feature gin_trgm_ops);
"""

# Tambahan: Modifikasi tabel checkpoints untuk menambah kolom user_id
CHECKPOINTS_USER_ID_SQL = """
-- Tambah kolom user_id ke checkpoints jika belum ada
ALTER TABLE checkpoints 
ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);

-- Buat index untuk filter by user_id
CREATE INDEX IF NOT EXISTS idx_checkpoints_user_id 
ON checkpoints (user_id);

-- Index gabungan untuk query yang sering: filter by user_id + thread_id
CREATE INDEX IF NOT EXISTS idx_checkpoints_user_thread 
ON checkpoints (user_id, thread_id);
"""

SAVE_PRD_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION save_prd_tx(
    p_id UUID DEFAULT NULL,
    p_user_id VARCHAR(255) DEFAULT NULL,
    p_feature TEXT DEFAULT NULL,
    p_introduction TEXT DEFAULT NULL,
    p_user_stories TEXT DEFAULT NULL,
    p_functional_requirements TEXT DEFAULT NULL,
    p_non_functional_requirements TEXT DEFAULT NULL,
    p_assumptions TEXT DEFAULT NULL,
    p_dependencies TEXT DEFAULT NULL,
    p_risks_and_mitigations TEXT DEFAULT NULL,
    p_timeline TEXT DEFAULT NULL,
    p_stakeholders TEXT DEFAULT NULL,
    p_metrics TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    v_id UUID;
    v_version INTEGER;
BEGIN
    IF p_user_id IS NULL OR p_user_id = '' THEN
        RAISE EXCEPTION 'User ID harus diisi (tidak boleh null atau empty)';
    END IF;

    IF p_feature IS NULL OR p_feature = '' THEN
        RAISE EXCEPTION 'Feature name harus diisi (tidak boleh null atau empty)';
    END IF;

    IF p_id IS NOT NULL THEN
        SELECT version INTO v_version FROM prds WHERE id = p_id AND user_id = p_user_id;
        IF FOUND THEN
            v_version := v_version + 1;
        ELSE
            v_version := 1;
        END IF;
    ELSE
        v_version := 1;
    END IF;

    INSERT INTO prds (
        id, user_id, feature, introduction, user_stories, functional_requirements,
        non_functional_requirements, assumptions, dependencies,
        risks_and_mitigations, timeline, stakeholders, metrics, version
    )
    VALUES (
        COALESCE(p_id, gen_random_uuid()), p_user_id, p_feature, p_introduction, p_user_stories,
        p_functional_requirements, p_non_functional_requirements, p_assumptions,
        p_dependencies, p_risks_and_mitigations, p_timeline, p_stakeholders,
        p_metrics, v_version
    )
    ON CONFLICT (id) DO UPDATE SET
        user_id = EXCLUDED.user_id,
        feature = EXCLUDED.feature,
        introduction = EXCLUDED.introduction,
        user_stories = EXCLUDED.user_stories,
        functional_requirements = EXCLUDED.functional_requirements,
        non_functional_requirements = EXCLUDED.non_functional_requirements,
        assumptions = EXCLUDED.assumptions,
        dependencies = EXCLUDED.dependencies,
        risks_and_mitigations = EXCLUDED.risks_and_mitigations,
        timeline = EXCLUDED.timeline,
        stakeholders = EXCLUDED.stakeholders,
        metrics = EXCLUDED.metrics,
        version = EXCLUDED.version,
        updated_at = NOW()
    WHERE prds.user_id = p_user_id
    RETURNING id INTO v_id;

    IF v_id IS NULL THEN
        RAISE EXCEPTION 'Tidak dapat update PRD: user_id tidak cocok';
    END IF;

    RETURN v_id;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Transaction rolled back: %', SQLERRM;
    RAISE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
"""


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding.embed_documents(texts)

async def initialize_resource(resource: Any) -> None:
    """
    Run setup() for resources that support it; tolerate repeated invocations.
    """
    if resource is None:
        return

    setup = getattr(resource, "setup", None)
    if setup is None:
        return

    try:
        await setup()
    except Exception as exc:
        message = str(exc).lower()
        if "duplicate" not in message and "already exists" not in message:
            raise


async def run_sql(db_uri: str, statement: str) -> None:
    conn = await AsyncConnection.connect(db_uri)
    try:
        async with conn.cursor() as cur:
            await cur.execute(statement)
        await conn.commit()
    finally:
        await conn.close()


async def setup_prd_schema(db_uri: str) -> None:
    await run_sql(db_uri, PRD_TABLE_SQL)
    await run_sql(db_uri, SAVE_PRD_FUNCTION_SQL)


async def main() -> None:
    load_dotenv()
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise RuntimeError("DB_URI not configured")

    await setup_prd_schema(db_uri)

    # Setup checkpointer dulu (biar tabel checkpoints dibuat)
    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        await initialize_resource(checkpointer)

    # BARU tambahkan kolom user_id setelah tabel checkpoints ada
    await run_sql(db_uri, CHECKPOINTS_USER_ID_SQL)

    async with AsyncPostgresStore.from_conn_string(
        db_uri,
        index={"dims": 1536, "embed": embed_texts},
    ) as store:
        await initialize_resource(store)

    print("Database resources are ready.")


if __name__ == "__main__":
    asyncio.run(main())