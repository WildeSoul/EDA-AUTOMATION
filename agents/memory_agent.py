"""
Memory Agent — Phase 3.

Queries the MemoryStore for semantically similar datasets, filters out
low-confidence matches, aggregates historical knowledge, and formats
explainability reasons for the dashboard.
"""

from __future__ import annotations
import logging
from typing import Any

from agents import BaseAgent, register_agent
from data.memory_store import MemoryStore
from config.settings import MEMORY_INJECTION_THRESHOLD

logger = logging.getLogger(__name__)

@register_agent
class MemoryAgent(BaseAgent):
    """Retrieves and filters cross-session memory to inject historical context."""

    name: str = "memory"

    def __init__(self, db_dir: str | None = None) -> None:
        """Initialise MemoryAgent with a MemoryStore instance.

        Args:
            db_dir: Optional custom database path for ChromaDB.
        """
        self.model = "nomic-embed-text:latest"
        self._store = MemoryStore(db_dir=db_dir)

    def run(self, intelligence: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Execute semantic search and aggregate matching session memories.

        Args:
            intelligence: Current dataset intelligence payload.
            **kwargs: Extra parameters like limit (default 3) and custom threshold.

        Returns:
            Structured agent result dict containing similar datasets and aggregated memories.
        """
        limit = kwargs.get("limit", 3)
        threshold = kwargs.get("threshold", MEMORY_INJECTION_THRESHOLD)
        
        try:
            # Step 1: Find similar dataset profiles (includes fusion scoring & reasons)
            matches = self._store.search_similar_datasets(intelligence, limit=limit)
            
            similar_datasets = []
            injected_memories = {
                "insights": [],
                "hypotheses": [],
                "cleaning_actions": []
            }
            evidence = []

            for match in matches:
                rank_score = match["rank_score"]
                fingerprint = match["fingerprint"]
                
                # Check confidence gate threshold
                if rank_score < threshold:
                    logger.info(
                        "Memory Confidence Gate: Ignored dataset '%s' with low rank_score: %.2f (threshold=%.2f)",
                        match["dataset"], rank_score, threshold
                    )
                    continue

                # Increment reuse count for chosen dataset memories
                self._store.increment_reuse_count(fingerprint)

                similar_datasets.append({
                    "dataset": match["dataset"],
                    "similarity": match["similarity"],
                    "rank_score": rank_score,
                    "reasons": match["reasons"],
                    "fingerprint": fingerprint
                })

                # Fetch matching insights, hypotheses, and cleaning steps
                dataset_mems = self._store.retrieve_dataset_memory(fingerprint)
                
                # Add source dataset tagging to memory items for clarity
                for ins in dataset_mems["insights"]:
                    ins["source_dataset"] = match["dataset"]
                    injected_memories["insights"].append(ins)
                    
                for hyp in dataset_mems["hypotheses"]:
                    hyp["source_dataset"] = match["dataset"]
                    injected_memories["hypotheses"].append(hyp)
                    
                for clean in dataset_mems["cleaning_actions"]:
                    clean["source_dataset"] = match["dataset"]
                    injected_memories["cleaning_actions"].append(clean)

                evidence.append(f"Retrieved memories from: {match['dataset']} (Rank score: {rank_score})")

            # Format the output result structure
            res_payload = {
                "similar_datasets": similar_datasets,
                "injected_memories": injected_memories
            }

            return self._make_result(
                success=True,
                result=res_payload,
                evidence=evidence,
                confidence=1.0 if similar_datasets else 0.0
            )

        except Exception as exc:
            logger.error("MemoryAgent execution failed: %s", exc)
            return self._make_result(
                success=False,
                error=str(exc)
            )
