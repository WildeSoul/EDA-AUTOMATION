"""
Plan Validator Layer — Phase 2.

Validates that LLM-generated execution plans are structural, dependency-complete,
acyclic, and reference valid agents.
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Valid agent names in the platform
VALID_AGENTS = {"schema", "quality", "hypothesis", "eda", "insight"}

def validate_plan(plan: dict[str, Any]) -> list[str]:
    """Validate an execution plan for syntax, cycle safety, and dependency completeness.

    Args:
        plan: The execution plan dictionary.

    Returns:
        A list of error message strings. If empty, the plan is valid.
    """
    errors: list[str] = []

    if not isinstance(plan, dict):
        return ["Plan must be a JSON object (dict)."]

    steps = plan.get("steps")
    if not isinstance(steps, list):
        return ["Plan must contain a 'steps' list."]

    if not steps:
        return ["Plan steps list cannot be empty."]

    # 1. Check duplicate step IDs & duplicate steps
    seen_ids = set()
    step_id_map = {}
    
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"Step at index {i} is not a valid JSON object.")
            continue

        step_id = step.get("id")
        if not step_id:
            errors.append(f"Step at index {i} is missing 'id'.")
            continue

        if step_id in seen_ids:
            errors.append(f"Duplicate step ID found: '{step_id}'.")
        seen_ids.add(step_id)
        step_id_map[step_id] = step

    if errors:
        return errors  # Stop early if basic structure is broken

    # Graph for cycle detection: node -> list of dependencies (incoming edges)
    # i.e., step_id depends on these parent step_ids.
    graph: dict[str, list[str]] = {}

    for step_id, step in step_id_map.items():
        agent = step.get("agent")
        depends_on = step.get("depends_on", [])
        input_mapping = step.get("input_mapping", {})

        # 2. Check valid agent names
        if agent not in VALID_AGENTS:
            errors.append(f"Step '{step_id}' references invalid agent name: '{agent}'.")

        # 3. Check depends_on structure
        if not isinstance(depends_on, list):
            errors.append(f"Step '{step_id}' 'depends_on' must be a list.")
            depends_on = []

        # 4. Check missing dependencies
        valid_deps = []
        for dep in depends_on:
            if dep not in step_id_map:
                errors.append(f"Step '{step_id}' depends on non-existent step: '{dep}'.")
            else:
                valid_deps.append(dep)
        graph[step_id] = valid_deps

        # 5. Check invalid input mappings (e.g. mapping inputs from steps not in depends_on)
        if not isinstance(input_mapping, dict):
            errors.append(f"Step '{step_id}' 'input_mapping' must be a dict.")
        else:
            for param, source_step_id in input_mapping.items():
                if source_step_id not in depends_on:
                    errors.append(
                        f"Step '{step_id}' maps input param '{param}' to step '{source_step_id}' "
                        f"which is not listed in its 'depends_on'."
                    )

    # 6. Check for dependency cycles (circular dependencies) using DFS
    visited: dict[str, int] = {node: 0 for node in graph}  # 0=unvisited, 1=visiting, 2=visited

    def has_cycle(node: str) -> bool:
        visited[node] = 1  # visiting
        for neighbor in graph[node]:
            if visited[neighbor] == 1:
                return True  # Found cycle
            elif visited[neighbor] == 0:
                if has_cycle(neighbor):
                    return True
        visited[node] = 2  # visited
        return False

    for node in graph:
        if visited[node] == 0:
            if has_cycle(node):
                errors.append("Circular dependency cycle detected in execution plan.")
                break

    return errors
