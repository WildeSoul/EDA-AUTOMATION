"""
Natural Language to SQL — converts user questions into DuckDB SQL queries.

Uses qwen3-coder-next:cloud for SQL generation.
Integrates with Dataset Intelligence for semantic column awareness.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TYPE_CHECKING

import pandas as pd

from config.settings import AGENT_MODELS, TIMEOUT_CLOUD
from core.llm import query_ollama

if TYPE_CHECKING:
    from data.engine import DuckDBEngine

logger = logging.getLogger(__name__)


def _extract_sql(raw: str) -> str:
    """Extract a SQL query from LLM output, handling markdown code blocks.

    Args:
        raw: Raw LLM response text.

    Returns:
        Cleaned SQL string.
    """
    text = raw.strip()

    # Try extracting from ```sql ... ``` blocks
    sql_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()

    # Try extracting from ``` ... ``` blocks
    code_match = re.search(r"```\s*\n?(.*?)```", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # If it starts with SELECT, take it as-is
    if text.upper().startswith("SELECT"):
        # Take up to the first semicolon or end
        end = text.find(";")
        return text[:end] if end > 0 else text

    # Last resort: find SELECT in the text
    select_match = re.search(r"(SELECT\s.+?)(?:;|\Z)", text, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()

    return text


def generate_sql(
    question: str,
    table_schema: list[dict[str, Any]],
    table_name: str = "dataset",
    intelligence: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Generate a DuckDB SQL query from a natural language question.

    Args:
        question: The user's question in plain English.
        table_schema: Column definitions from DuckDB DESCRIBE.
        table_name: Name of the registered table.
        intelligence: Optional Dataset Intelligence dict for semantic context.

    Returns:
        {"sql": str, "explanation": str}
    """
    model = AGENT_MODELS["coding"]

    # Build schema description
    schema_lines = []
    for col_info in table_schema:
        col_name = col_info.get("column_name", col_info.get("Field", ""))
        col_type = col_info.get("column_type", col_info.get("Type", ""))
        line = f"  - {col_name} ({col_type})"

        # Add semantic description from intelligence if available
        if intelligence and col_name in intelligence.get("columns", {}):
            desc = intelligence["columns"][col_name].get("description", "")
            sem_type = intelligence["columns"][col_name].get("semantic_type", "")
            if desc:
                line += f"  -- {desc}"
            elif sem_type:
                line += f"  -- type: {sem_type}"

        schema_lines.append(line)

    schema_text = "\n".join(schema_lines)

    # Add dataset context if available
    context_text = ""
    if intelligence:
        context_text = (
            f"\nDataset type: {intelligence.get('dataset_type', 'unknown')}\n"
            f"Domain: {intelligence.get('domain', 'unknown')}\n"
            f"Rows: {intelligence.get('row_count', 'unknown')}\n"
        )

    prompt = f"""Convert this question into a DuckDB SQL query.

TABLE: {table_name}
COLUMNS:
{schema_text}
{context_text}
QUESTION: {question}

RULES:
1. Return ONLY a valid DuckDB SELECT query
2. Use the table name "{table_name}" exactly
3. Add LIMIT 1000 if the query doesn't already have a LIMIT
4. Use proper DuckDB SQL dialect
5. Handle NULL values appropriately
6. For aggregations, use meaningful column aliases
7. Order results logically (e.g., by count DESC for frequency queries)

After the SQL query, add a brief one-line explanation starting with "Explanation:".

SQL:"""

    system = (
        "You are a SQL expert. Generate precise DuckDB SQL queries from natural language. "
        "Return the SQL query followed by a brief explanation. "
        "Never use write operations (INSERT, UPDATE, DELETE, DROP)."
    )

    try:
        raw = query_ollama(model, prompt, system=system, timeout=TIMEOUT_CLOUD)
        logger.debug("NL→SQL raw response: %s", raw[:300])

        # Split SQL from explanation
        sql = _extract_sql(raw)
        explanation = ""
        for line in raw.split("\n"):
            if line.strip().lower().startswith("explanation:"):
                explanation = line.strip()[len("Explanation:"):].strip()
                break

        if not explanation:
            explanation = f"Generated SQL for: {question}"

        logger.info("Generated SQL: %s", sql[:100])
        return {"sql": sql, "explanation": explanation}

    except Exception as exc:
        logger.error("SQL generation failed: %s", exc)
        return {
            "sql": "",
            "explanation": f"Failed to generate SQL: {exc}",
        }


def execute_nl_query(
    question: str,
    engine: DuckDBEngine,
    table_name: str = "dataset",
    intelligence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """End-to-end: natural language question → SQL → execute → results.

    Args:
        question: The user's question in plain English.
        engine: An initialized DuckDBEngine with a registered table.
        table_name: Name of the table to query.
        intelligence: Optional Dataset Intelligence dict.

    Returns:
        {
            "question": str,
            "sql": str,
            "explanation": str,
            "result": DataFrame or None,
            "row_count": int,
            "columns": list[str],
            "success": bool,
            "error": str | None
        }
    """
    result: dict[str, Any] = {
        "question": question,
        "sql": "",
        "explanation": "",
        "result": None,
        "row_count": 0,
        "columns": [],
        "success": False,
        "error": None,
    }

    # Step 1: Get table schema
    try:
        schema = engine.get_schema(table_name)
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    # Step 2: Generate SQL
    gen = generate_sql(question, schema, table_name, intelligence)
    result["sql"] = gen["sql"]
    result["explanation"] = gen["explanation"]

    if not gen["sql"]:
        result["error"] = "Could not generate SQL from the question"
        return result

    # Step 3: Execute
    try:
        df_result = engine.query(gen["sql"])
        result["result"] = df_result
        result["row_count"] = len(df_result)
        result["columns"] = df_result.columns.tolist()
        result["success"] = True
        logger.info(
            "NL query executed: '%s' → %d rows",
            question[:50], len(df_result),
        )
    except ValueError as exc:
        result["error"] = str(exc)
        logger.error("NL query execution failed: %s", exc)

    return result
