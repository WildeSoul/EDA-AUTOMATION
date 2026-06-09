"""
Agent system for AI Data Intelligence Platform.

Phase 1 agents:
- SchemaAgent: Semantic column understanding
- QualityAgent: Data quality diagnostics
- HypothesisAgent: Generate testable hypotheses
- EDAAgent: Statistical analysis (wraps existing modules)
- InsightAgent: Executive synthesis with explainability
"""
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from config.settings import AGENT_CACHE_TTL

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Agent Cache Layer
# Before every LLM call: Request → Cache Check → LLM
# ──────────────────────────────────────────────
_agent_cache: dict[str, tuple[float, dict]] = {}


def _cache_key(agent_name: str, input_hash: str) -> str:
    """Generate a unique cache key for an agent + input combination."""
    return f"{agent_name}:{input_hash}"


def cache_get(agent_name: str, input_hash: str) -> dict | None:
    """Check cache for a previous agent result. Returns None if miss or expired."""
    key = _cache_key(agent_name, input_hash)
    if key in _agent_cache:
        timestamp, result = _agent_cache[key]
        if time.time() - timestamp < AGENT_CACHE_TTL:
            logger.info(f"Cache HIT for {agent_name} (hash={input_hash[:8]}...)")
            return result
        else:
            del _agent_cache[key]
            logger.debug(f"Cache EXPIRED for {agent_name}")
    return None


def cache_set(agent_name: str, input_hash: str, result: dict) -> None:
    """Store an agent result in cache."""
    key = _cache_key(agent_name, input_hash)
    _agent_cache[key] = (time.time(), result)
    logger.debug(f"Cache SET for {agent_name} (hash={input_hash[:8]}...)")


def compute_input_hash(*args: Any) -> str:
    """Compute MD5 hash of agent inputs for cache lookup."""
    raw = json.dumps(args, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


# ──────────────────────────────────────────────
# Base Agent Class
# ──────────────────────────────────────────────
class BaseAgent(ABC):
    """Abstract base class for all agents.

    Every agent:
    - Has a name and a model assignment
    - Takes the intelligence dict as primary input
    - Returns structured results with explainability metadata
    - Supports caching via dataset fingerprint
    """

    name: str = "base"
    model: str = ""

    @abstractmethod
    def run(self, intelligence: dict, **kwargs) -> dict:
        """Execute the agent's analysis.

        Args:
            intelligence: The Dataset Intelligence dict (the brain).
            **kwargs: Agent-specific parameters.

        Returns:
            dict with at minimum:
            {
                "agent": self.name,
                "success": bool,
                "result": {...},        # Agent-specific output
                "evidence": [...],      # Supporting data points
                "confidence": float,    # 0-1 overall confidence
                "error": str | None,
            }
        """
        raise NotImplementedError

    def run_cached(self, intelligence: dict, **kwargs) -> dict:
        """Run with cache layer. Uses dataset fingerprint for cache key."""
        fingerprint = intelligence.get("dataset_fingerprint", "")
        extra_hash = compute_input_hash(kwargs) if kwargs else ""
        input_hash = compute_input_hash(fingerprint, self.name, extra_hash)

        cached = cache_get(self.name, input_hash)
        if cached is not None:
            # Make a copy to avoid mutating cache entries globally
            res = dict(cached)
            res["cached"] = True
            res["duration_sec"] = 0.0
            return res

        start_time = time.time()
        result = self.run(intelligence, **kwargs)
        elapsed = time.time() - start_time
        
        result["duration_sec"] = round(elapsed, 3)
        result["model"] = self.model
        result["cached"] = False

        if result.get("success", False):
            cache_set(self.name, input_hash, result)
        return result

    def _make_result(
        self,
        success: bool,
        result: dict | None = None,
        evidence: list | None = None,
        confidence: float = 0.0,
        error: str | None = None,
    ) -> dict:
        """Helper to build a standard agent result dict."""
        return {
            "agent": self.name,
            "success": success,
            "result": result or {},
            "evidence": evidence or [],
            "confidence": confidence,
            "error": error,
            "model": self.model,
            "duration_sec": 0.0,
            "cached": False,
        }


# Agent registry — populated by imports
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    """Decorator to register an agent class in the global registry."""
    AGENT_REGISTRY[cls.name] = cls
    return cls
