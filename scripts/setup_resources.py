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

CREATE TABLE IF NOT EXISTS prds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature TEXT NOT NULL,
    introduction TEXT,
    user_stories JSONB,
    functional_requirements JSONB,
    non_functional_requirements JSONB,
    assumptions JSONB,
    dependencies JSONB,
    risks_and_mitigations JSONB,
    timeline TEXT,
    stakeholders JSONB,
    metrics JSONB,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prds_feature ON prds (feature);
CREATE INDEX IF NOT EXISTS idx_prds_user_stories ON prds USING GIN (user_stories);
"""

SAVE_PRD_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION save_prd_tx(
    p_id UUID DEFAULT NULL,
    p_feature TEXT DEFAULT NULL,
    p_introduction TEXT DEFAULT NULL,
    p_user_stories JSONB DEFAULT NULL,
    p_functional_requirements JSONB DEFAULT NULL,
    p_non_functional_requirements JSONB DEFAULT NULL,
    p_assumptions JSONB DEFAULT NULL,
    p_dependencies JSONB DEFAULT NULL,
    p_risks_and_mitigations JSONB DEFAULT NULL,
    p_timeline TEXT DEFAULT NULL,
    p_stakeholders JSONB DEFAULT NULL,
    p_metrics JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    v_id UUID;
    v_version INTEGER;
BEGIN
    IF p_feature IS NULL OR p_feature = '' THEN
        RAISE EXCEPTION 'Feature name harus diisi (tidak boleh null atau empty)';
    END IF;

    IF p_id IS NOT NULL THEN
        SELECT version INTO v_version FROM prds WHERE id = p_id;
        IF FOUND THEN
            v_version := v_version + 1;
        ELSE
            v_version := 1;
        END IF;
    ELSE
        v_version := 1;
    END IF;

    INSERT INTO prds (
        id, feature, introduction, user_stories, functional_requirements,
        non_functional_requirements, assumptions, dependencies,
        risks_and_mitigations, timeline, stakeholders, metrics, version
    )
    VALUES (
        COALESCE(p_id, gen_random_uuid()), p_feature, p_introduction, p_user_stories,
        p_functional_requirements, p_non_functional_requirements, p_assumptions,
        p_dependencies, p_risks_and_mitigations, p_timeline, p_stakeholders,
        p_metrics, v_version
    )
    ON CONFLICT (id) DO UPDATE SET
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
    RETURNING id INTO v_id;

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

    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        await initialize_resource(checkpointer)

    async with AsyncPostgresStore.from_conn_string(
        db_uri,
        index={"dims": 1536, "embed": embed_texts},
    ) as store:
        await initialize_resource(store)

    print("Database resources are ready.")


if __name__ == "__main__":
    asyncio.run(main())
