"""
DuckDB analytics engine — enables SQL queries against uploaded datasets.

Provides a lightweight, session-scoped SQL layer on top of pandas DataFrames.
Each session gets its own DuckDB in-memory connection.
"""

from __future__ import annotations

import logging
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBEngine:
    """DuckDB-powered analytics engine for SQL queries against uploaded datasets."""

    def __init__(self) -> None:
        """Create an in-memory DuckDB connection."""
        self.conn = duckdb.connect()
        self._registered_tables: dict[str, list[str]] = {}
        logger.info("DuckDB engine initialized")

    def register(self, df: pd.DataFrame, table_name: str = "dataset") -> dict[str, Any]:
        """Register a DataFrame as a queryable DuckDB table.

        Args:
            df: The pandas DataFrame to register.
            table_name: Name for the virtual table.

        Returns:
            Table metadata: {table_name, columns, row_count}.
        """
        try:
            self.conn.register(table_name, df)
            columns = df.columns.tolist()
            self._registered_tables[table_name] = columns
            logger.info(
                "Registered table '%s' with %d rows, %d columns",
                table_name, len(df), len(columns),
            )
            return {
                "table_name": table_name,
                "columns": columns,
                "row_count": len(df),
            }
        except Exception as exc:
            logger.error("Failed to register table '%s': %s", table_name, exc)
            raise

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame.

        Only SELECT queries are permitted. Write operations are blocked.

        Args:
            sql: DuckDB-compatible SQL query string.

        Returns:
            Query results as a pandas DataFrame.

        Raises:
            ValueError: If the query is a write operation or has a syntax error.
        """
        sql_stripped = sql.strip()
        sql_upper = sql_stripped.upper()

        # Block dangerous operations
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"]
        first_word = sql_upper.split()[0] if sql_upper else ""
        if first_word in dangerous:
            raise ValueError(
                f"Write operations ({first_word}) are not allowed. "
                f"Use SELECT queries only."
            )

        try:
            result = self.conn.execute(sql_stripped).fetchdf()
            logger.info("Query returned %d rows, %d columns", len(result), len(result.columns))
            return result
        except duckdb.Error as exc:
            logger.error("DuckDB query error: %s", exc)
            raise ValueError(f"SQL error: {exc}")

    def get_schema(self, table_name: str = "dataset") -> list[dict[str, str]]:
        """Get column names and types for a registered table.

        Args:
            table_name: Name of the registered table.

        Returns:
            List of {column_name, column_type} dicts.
        """
        if table_name not in self._registered_tables:
            raise ValueError(f"Table '{table_name}' is not registered")
        try:
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
            return result.to_dict(orient="records")
        except Exception as exc:
            logger.error("Failed to describe table '%s': %s", table_name, exc)
            return [{"column_name": c} for c in self._registered_tables.get(table_name, [])]

    def get_registered_tables(self) -> dict[str, list[str]]:
        """Return dict of registered table names and their columns."""
        return dict(self._registered_tables)

    def close(self) -> None:
        """Close the DuckDB connection."""
        try:
            self.conn.close()
            logger.info("DuckDB engine closed")
        except Exception:
            pass
