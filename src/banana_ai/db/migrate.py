"""Safe additive database migration for Banana AI feature upgrade.

This script adds new columns to existing tables and creates the new
prediction_feedback table.  It NEVER drops tables or existing columns.
It is idempotent: safe to run multiple times.

Run with:
    python -m banana_ai.db.migrate
"""

import logging
from sqlalchemy import text, inspect

from banana_ai.db.session import Base, engine
from banana_ai.db import models  # noqa: F401 — registers all models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _column_exists(inspector, table: str, column: str) -> bool:
    cols = [c["name"] for c in inspector.get_columns(table)]
    return column in cols


def _table_exists(inspector, table: str) -> bool:
    return table in inspector.get_table_names()


def run_migration():
    """Execute safe, additive schema changes."""
    inspector = inspect(engine)

    with engine.begin() as conn:
        # ------------------------------------------------------------------ #
        # observations table — add new columns if missing                     #
        # ------------------------------------------------------------------ #
        if _table_exists(inspector, "observations"):
            if not _column_exists(inspector, "observations", "storage_condition"):
                conn.execute(
                    text("ALTER TABLE observations ADD COLUMN storage_condition VARCHAR(64)")
                )
                logger.info("Added column: observations.storage_condition")
            if not _column_exists(inspector, "observations", "input_method"):
                conn.execute(
                    text("ALTER TABLE observations ADD COLUMN input_method VARCHAR(32)")
                )
                logger.info("Added column: observations.input_method")

        # ------------------------------------------------------------------ #
        # predictions table — add new columns if missing                      #
        # ------------------------------------------------------------------ #
        if _table_exists(inspector, "predictions"):
            for col, col_type in [
                ("estimated_min_days", "INTEGER"),
                ("estimated_max_days", "INTEGER"),
                ("shelf_life_method", "VARCHAR(64)"),
            ]:
                if not _column_exists(inspector, "predictions", col):
                    conn.execute(
                        text(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
                    )
                    logger.info("Added column: predictions.%s", col)

        # ------------------------------------------------------------------ #
        # Create any new tables (including prediction_feedback)               #
        # ------------------------------------------------------------------ #
        Base.metadata.create_all(bind=conn)
        logger.info("Schema migration complete (create_all for new tables).")

    logger.info("Migration finished successfully.")


if __name__ == "__main__":
    run_migration()
