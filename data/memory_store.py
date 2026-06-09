"""
Memory Store — Phase 3.

Wraps ChromaDB client to manage persistence, semantic retrieval,
score fusion ranking, explainability metadata, and memory pruning.
"""

from __future__ import annotations
import os
import math
import uuid
import logging
from datetime import datetime, timezone
from typing import Any

import chromadb
from core.embeddings import get_nomic_embedding
from config.settings import SESSION_DB_PATH

logger = logging.getLogger(__name__)

class MemoryStore:
    """Manages ChromaDB vector collections for persistent cross-session memory."""

    def __init__(self, db_dir: str | None = None) -> None:
        """Initialise ChromaDB persistent client and retrieve/create collections.

        Args:
            db_dir: Filesystem path to store ChromaDB collections.
                    Defaults to data/chroma_db/ relative to project root.
        """
        if db_dir is None:
            db_dir = os.environ.get("TESTING_CHROMA_DIR")

        if db_dir is None:
            # Base it on the directory containing sessions.db
            base_dir = os.path.dirname(SESSION_DB_PATH)
            self._db_dir = os.path.join(base_dir, "chroma_db")
        else:
            self._db_dir = db_dir

        try:
            os.makedirs(self._db_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._db_dir)
            
            # Setup collections with cosine similarity distance metric
            self.col_datasets = self._client.get_or_create_collection(
                name="datasets", metadata={"hnsw:space": "cosine"}
            )
            self.col_insights = self._client.get_or_create_collection(
                name="insights", metadata={"hnsw:space": "cosine"}
            )
            self.col_hypotheses = self._client.get_or_create_collection(
                name="hypotheses", metadata={"hnsw:space": "cosine"}
            )
            self.col_cleaning = self._client.get_or_create_collection(
                name="cleaning_actions", metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB MemoryStore initialised at: %s", self._db_dir)
        except Exception as exc:
            logger.error("Failed to initialise ChromaDB MemoryStore: %s", exc)
            raise

    def close(self) -> None:
        """Close the underlying ChromaDB persistent client and free resources."""
        if hasattr(self, "_client") and hasattr(self._client, "close"):
            try:
                self._client.close()
                logger.info("ChromaDB MemoryStore connection closed.")
            except Exception as exc:
                logger.debug("Error closing ChromaDB: %s", exc)

    # ── Insertion API ────────────────────────────────────────────────────────

    def add_dataset_profile(
        self,
        sid: str,
        fingerprint: str,
        filename: str,
        intelligence: dict[str, Any],
        version: int
    ) -> None:
        """Insert or update a dataset's profile summary and metadata."""
        try:
            # Construct summary document
            dataset_type = intelligence.get("dataset_type", "unknown")
            domain = intelligence.get("domain", "unknown")
            columns = intelligence.get("columns", {})
            col_list = list(columns.keys())
            
            summary_doc = (
                f"Dataset Profile: {filename}\n"
                f"Domain: {domain}\n"
                f"Dataset Type: {dataset_type}\n"
                f"Columns: {', '.join(col_list)}\n"
                f"Row Count: {intelligence.get('row_count', 0)}, Column Count: {len(col_list)}"
            )

            embedding = get_nomic_embedding(summary_doc)
            
            metadata = {
                "sid": sid,
                "fingerprint": fingerprint,
                "filename": filename,
                "dataset_type": dataset_type,
                "domain": domain,
                "version": version,
                "row_count": intelligence.get("row_count", 0),
                "col_count": len(col_list),
                "quality_score": float(intelligence.get("quality_score", 1.0)),
                "reuse_count": 0,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            self.col_datasets.upsert(
                ids=[fingerprint],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[summary_doc]
            )
            logger.info("Dataset profile stored in ChromaDB: %s (v%d)", filename, version)
        except Exception as exc:
            logger.error("Failed to add dataset profile to ChromaDB: %s", exc)

    def add_insight(
        self,
        sid: str,
        fingerprint: str,
        agent: str,
        title: str,
        content: str,
        confidence: float
    ) -> None:
        """Store a synthesized insight in the insights collection."""
        try:
            ins_id = f"{fingerprint}_insight_{uuid.uuid4().hex[:8]}"
            embedding = get_nomic_embedding(content)
            
            metadata = {
                "sid": sid,
                "dataset_fingerprint": fingerprint,
                "agent": agent,
                "title": title,
                "confidence": float(confidence),
                "reuse_count": 0,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            self.col_insights.add(
                ids=[ins_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[content]
            )
            logger.info("Insight stored in ChromaDB: %s", title)
        except Exception as exc:
            logger.error("Failed to add insight to ChromaDB: %s", exc)

    def add_hypothesis(
        self,
        sid: str,
        fingerprint: str,
        hypothesis: str,
        status: str,
        evidence_score: float,
        confidence: float
    ) -> None:
        """Store a formulated hypothesis in the hypotheses collection."""
        try:
            hyp_id = f"{fingerprint}_hypothesis_{uuid.uuid4().hex[:8]}"
            embedding = get_nomic_embedding(hypothesis)
            
            metadata = {
                "sid": sid,
                "dataset_fingerprint": fingerprint,
                "hypothesis": hypothesis,
                "status": status,
                "evidence_score": float(evidence_score),
                "confidence": float(confidence),
                "reuse_count": 0,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            self.col_hypotheses.add(
                ids=[hyp_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[hypothesis]
            )
            logger.info("Hypothesis stored in ChromaDB: %s", hypothesis[:40])
        except Exception as exc:
            logger.error("Failed to add hypothesis to ChromaDB: %s", exc)

    def add_cleaning_action(
        self,
        sid: str,
        fingerprint: str,
        column: str,
        action_type: str,
        description: str
    ) -> None:
        """Store a data cleaning action in the cleaning_actions collection."""
        try:
            clean_id = f"{fingerprint}_clean_{uuid.uuid4().hex[:8]}"
            embedding = get_nomic_embedding(description)
            
            metadata = {
                "sid": sid,
                "dataset_fingerprint": fingerprint,
                "column": column,
                "action_type": action_type,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            self.col_cleaning.add(
                ids=[clean_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[description]
            )
            logger.info("Cleaning action stored in ChromaDB for column: %s", column)
        except Exception as exc:
            logger.error("Failed to add cleaning action to ChromaDB: %s", exc)

    # ── Search & Retrieval API ───────────────────────────────────────────────

    def search_similar_datasets(
        self,
        intelligence: dict[str, Any],
        limit: int = 3
    ) -> list[dict[str, Any]]:
        """Search ChromaDB for semantically similar dataset profiles and compute fusion scores.

        Args:
            intelligence: The intelligence payload of the current dataset.
            limit: Maximum matches to retrieve.

        Returns:
            List of retrieved dataset matches, each containing metadata, similarity score,
            fused rank score, and explainability reasons.
        """
        # Build query profile text
        filename = intelligence.get("filename", "")
        dataset_type = intelligence.get("dataset_type", "unknown")
        domain = intelligence.get("domain", "unknown")
        columns = intelligence.get("columns", {})
        col_list = list(columns.keys())
        
        query_doc = (
            f"Dataset Profile: {filename}\n"
            f"Domain: {domain}\n"
            f"Dataset Type: {dataset_type}\n"
            f"Columns: {', '.join(col_list)}\n"
            f"Row Count: {intelligence.get('row_count', 0)}, Column Count: {len(col_list)}"
        )

        query_embedding = get_nomic_embedding(query_doc)

        try:
            results = self.col_datasets.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        if not results or not results["metadatas"] or len(results["metadatas"][0]) == 0:
            return []

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]

        scored_matches = []
        for meta, dist, doc in zip(metadatas, distances, documents):
            # Cosine similarity calculation: cosine distance ranges from 0 to 2
            similarity = max(0.0, min(1.0, 1.0 - dist))
            
            # Retrieve weight inputs
            confidence = float(meta.get("quality_score", 1.0))
            reuse_count = float(meta.get("reuse_count", 0))
            created_at_str = meta.get("created_at", "")

            # Reuse score: min(1.0, ln(1 + reuse_count)/ln(10))
            reuse_score = min(1.0, math.log(1.0 + reuse_count) / math.log(10.0)) if reuse_count > 0 else 0.0

            # Recency: e^(-lambda * delta_days), lambda = 0.05
            try:
                created_dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)
                delta_days = max(0.0, (now_dt - created_dt).total_seconds() / 86400.0)
            except Exception:
                delta_days = 0.0
            recency = math.exp(-0.05 * delta_days)

            # Score Fusion Rank
            # Rank = 0.5 * Sim + 0.2 * Conf + 0.2 * Reuse + 0.1 * Recency
            rank_score = (0.5 * similarity) + (0.2 * confidence) + (0.2 * reuse_score) + (0.1 * recency)

            # Generate Explainability Reasons
            reasons = []
            
            # Domain check
            meta_domain = meta.get("domain", "unknown")
            if domain.lower() == meta_domain.lower() and domain.lower() != "unknown":
                reasons.append(f"same domain: {domain}")
            
            # Dataset Type check
            meta_type = meta.get("dataset_type", "unknown")
            if dataset_type.lower() == meta_type.lower() and dataset_type.lower() != "unknown":
                reasons.append(f"same target type: {dataset_type}")

            # Overlapping features check
            overlap_cols = [c for c in col_list if c in doc]
            if len(overlap_cols) > 0:
                # Include up to 3 overlapping columns in the reason description
                display_cols = ", ".join(overlap_cols[:2]) + ("..." if len(overlap_cols) > 2 else "")
                reasons.append(f"similar feature structure ({display_cols})")
            
            if not reasons:
                reasons.append("similar semantic profile")

            scored_matches.append({
                "dataset": meta.get("filename", "unknown"),
                "fingerprint": meta.get("fingerprint", ""),
                "similarity": round(similarity, 2),
                "rank_score": round(rank_score, 2),
                "reasons": reasons,
                "metadata": meta,
                "document": doc
            })

        # Sort by fused rank score descending
        scored_matches.sort(key=lambda x: x["rank_score"], reverse=True)
        return scored_matches

    def retrieve_dataset_memory(self, fingerprint: str) -> dict[str, Any]:
        """Fetch all insights, hypotheses, and cleaning actions associated with a dataset fingerprint."""
        memories = {
            "insights": [],
            "hypotheses": [],
            "cleaning_actions": []
        }

        try:
            # Query insights
            res_ins = self.col_insights.get(
                where={"dataset_fingerprint": fingerprint},
                include=["documents", "metadatas"]
            )
            if res_ins and res_ins["documents"]:
                for doc, meta in zip(res_ins["documents"], res_ins["metadatas"]):
                    memories["insights"].append({
                        "content": doc,
                        "title": meta.get("title", ""),
                        "agent": meta.get("agent", ""),
                        "confidence": meta.get("confidence", 1.0)
                    })

            # Query hypotheses
            res_hyp = self.col_hypotheses.get(
                where={"dataset_fingerprint": fingerprint},
                include=["documents", "metadatas"]
            )
            if res_hyp and res_hyp["documents"]:
                for doc, meta in zip(res_hyp["documents"], res_hyp["metadatas"]):
                    memories["hypotheses"].append({
                        "content": doc,
                        "hypothesis": meta.get("hypothesis", ""),
                        "status": meta.get("status", "untested"),
                        "evidence_score": meta.get("evidence_score", 0.0),
                        "confidence": meta.get("confidence", 1.0)
                    })

            # Query cleaning actions
            res_clean = self.col_cleaning.get(
                where={"dataset_fingerprint": fingerprint},
                include=["documents", "metadatas"]
            )
            if res_clean and res_clean["documents"]:
                for doc, meta in zip(res_clean["documents"], res_clean["metadatas"]):
                    memories["cleaning_actions"].append({
                        "description": doc,
                        "column": meta.get("column", ""),
                        "action_type": meta.get("action_type", "")
                    })

        except Exception as exc:
            logger.error("Failed to retrieve dataset memory: %s", exc)

        return memories

    def increment_reuse_count(self, fingerprint: str) -> None:
        """Increment the reuse count for a dataset profile to boost future ranking scores."""
        try:
            res = self.col_datasets.get(ids=[fingerprint])
            if res and res["metadatas"]:
                meta = res["metadatas"][0]
                meta["reuse_count"] = int(meta.get("reuse_count", 0)) + 1
                
                # Perform metadata-only update to prevent dimension mismatch of missing documents/embeddings
                self.col_datasets.update(
                    ids=[fingerprint],
                    metadatas=[meta]
                )
                logger.info("Incremented reuse count for dataset fingerprint: %s", fingerprint[:8])
        except Exception as exc:
            logger.error("Failed to increment dataset reuse count: %s", exc)

    def prune_memories(self, score_threshold: float = 0.5) -> int:
        """Remove low-scoring dataset profiles and their associated memories.

        Prunes datasets whose calculated rank score (assuming reuse=0 and similarity=1.0,
        simply based on recency decay and original quality score) falls below the threshold.

        Returns:
            Number of datasets pruned.
        """
        pruned_count = 0
        try:
            res = self.col_datasets.get(include=["metadatas"])
            if not res or not res["metadatas"]:
                return 0

            for meta in res["metadatas"]:
                fingerprint = meta.get("fingerprint", "")
                if not fingerprint:
                    continue

                # Compute baseline rank score decay: similarity=1.0, reuse=0
                confidence = float(meta.get("quality_score", 1.0))
                created_at_str = meta.get("created_at", "")

                try:
                    created_dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    now_dt = datetime.now(timezone.utc)
                    delta_days = max(0.0, (now_dt - created_dt).total_seconds() / 86400.0)
                except Exception:
                    delta_days = 0.0
                recency = math.exp(-0.05 * delta_days)

                # Rank under baseline: 0.5 * 1.0 + 0.2 * confidence + 0.1 * recency
                baseline_rank = 0.5 + (0.2 * confidence) + (0.1 * recency)

                if baseline_rank < score_threshold:
                    # Prune this dataset
                    self.col_datasets.delete(ids=[fingerprint])
                    
                    # Cascade delete related memories
                    self.col_insights.delete(where={"dataset_fingerprint": fingerprint})
                    self.col_hypotheses.delete(where={"dataset_fingerprint": fingerprint})
                    self.col_cleaning.delete(where={"dataset_fingerprint": fingerprint})
                    
                    pruned_count += 1
                    logger.info("Pruned stale dataset memory: %s", meta.get("filename"))

        except Exception as exc:
            logger.error("Failed to prune stale dataset memories: %s", exc)

        return pruned_count
