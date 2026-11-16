"""
Tests for UI Workflow, 4-Hub Structure, 17-Model Catalog, and E2E Pipeline Correctness.
Bridges the gap between backend regression tests and complete end-to-end workflow verification.
"""
import os
import sys
import json
import pytest

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from server import app, SESSIONS

client = TestClient(app)


# ═══════════════════════════════════════════════
# Test Suite 1: 17-Model Master Catalog UX
# ═══════════════════════════════════════════════

def test_api_models_catalog_integrity():
    """Verify that all 17 model options remain accessible, categorized by tier."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    
    assert "models" in data
    assert "default" in data
    models = data["models"]
    
    # Must have 17 models
    assert len(models) == 17, f"Expected 17 models, got {len(models)}"
    
    # Verify tier classifications
    tiers = {m["tier"] for m in models}
    assert "frontier" in tiers
    assert "specialist" in tiers
    assert "local" in tiers
    
    frontier_count = sum(1 for m in models if m["tier"] == "frontier")
    specialist_count = sum(1 for m in models if m["tier"] == "specialist")
    local_count = sum(1 for m in models if m["tier"] == "local")
    
    assert frontier_count == 4, f"Expected 4 Frontier models, got {frontier_count}"
    assert specialist_count >= 7, f"Expected at least 7 Specialist models, got {specialist_count}"
    assert local_count >= 4, f"Expected at least 4 Local models, got {local_count}"
    
    # Verify required keys per model item
    for m in models:
        assert "id" in m
        assert "name" in m
        assert "type" in m
        assert "tier" in m
        assert "available" in m


# ═══════════════════════════════════════════════
# Test Suite 2: End-to-End Workflow & 4-Hub Lifecycle
# ═══════════════════════════════════════════════

def test_e2e_demo_to_analysis_pipeline():
    """Verify full user journey: Load Demo -> Inspect -> Auto-EDA -> Stats -> Observability."""
    # 1. Ingest & Upload (Hub 1)
    load_resp = client.post("/api/demo/titanic")
    assert load_resp.status_code == 200
    load_data = load_resp.json()
    assert "session_id" in load_data
    sid = load_data["session_id"]
    assert load_data["rows"] > 0
    assert load_data["columns"] > 0
    
    # 2. Health & Schema Inspection (Hub 1)
    inspect_resp = client.get(f"/api/inspect/{sid}")
    assert inspect_resp.status_code == 200
    inspect_data = inspect_resp.json()
    assert inspect_data["rows"] == load_data["rows"]
    assert "data_health_score" in inspect_data
    assert inspect_data["data_health_score"] >= 0
    
    # 3. Profiling & Distributions (Hub 2)
    profile_resp = client.get(f"/api/profile/{sid}")
    assert profile_resp.status_code == 200
    profile_data = profile_resp.json()
    assert "numerical" in profile_data
    assert "categorical" in profile_data
    assert len(profile_data["numerical"]) > 0
    
    # 4. Feature Engineering & Stats (Hub 2)
    features_resp = client.post(f"/api/features/{sid}")
    assert features_resp.status_code == 200
    features_data = features_resp.json()
    assert "skewed_features" in features_data
    
    stats_resp = client.get(f"/api/stats/{sid}")
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert "pca" in stats_data or "normality" in stats_data
    
    # 5. ML & Smart Labs (Hub 3)
    anomaly_resp = client.post(f"/api/anomaly/{sid}?method=zscore&threshold=2.5")
    assert anomaly_resp.status_code == 200
    anomaly_data = anomaly_resp.json()
    assert "anomalous_count" in anomaly_data or "summary" in anomaly_data
    
    # 6. Observability Metrics (Hub 4)
    obs_resp = client.get("/api/observability/metrics")
    assert obs_resp.status_code == 200
    obs_data = obs_resp.json()
    assert "overall_success_rate" in obs_data
    assert "parallel_speedup" in obs_data


def test_session_persistence_rehydration():
    """Verify session can be reloaded and inspected after in-memory eviction."""
    # Create session
    resp = client.post("/api/demo/sales_transactions")
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    
    # Simulate in-memory eviction (e.g. server restart / worker recycle)
    if sid in SESSIONS:
        del SESSIONS[sid]
    
    # Rehydrate from SQLite session store via API
    reloaded_resp = client.get(f"/api/inspect/{sid}")
    assert reloaded_resp.status_code == 200
    reloaded_data = reloaded_resp.json()
    assert reloaded_data["rows"] > 0
    assert reloaded_data["filename"] == "sales_transactions.csv"


# ═══════════════════════════════════════════════
# Test Suite 3: Frontend Asset Integrity & Standards
# ═══════════════════════════════════════════════

def test_frontend_markup_and_routing_contracts():
    """Verify frontend HTML has all required hubs, mobile elements, and ARIA roles."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base_dir, "frontend", "index.html")
    assert os.path.exists(html_path)
    
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    
    # 1. 4-Hub Structure Verification
    assert 'data-hub="overview"' in html
    assert 'data-hub="analysis"' in html
    assert 'data-hub="ml"' in html
    assert 'data-hub="copilot"' in html
    
    # 2. Responsive Mobile Drawer & Bottom Navigation
    assert 'id="sidebar-backdrop"' in html
    assert 'id="btn-mobile-menu-toggle"' in html
    assert 'id="btn-sidebar-close"' in html
    assert 'id="mobile-bottom-nav"' in html
    
    # 3. Model Modal UX (Search + Tier Pills)
    assert 'id="model-modal"' in html
    assert 'id="model-search-input"' in html
    assert 'id="model-tier-tabs"' in html
    assert 'data-tier="frontier"' in html
    assert 'data-tier="specialist"' in html
    assert 'data-tier="local"' in html
    
    # 4. FOUT Prevention Script
    assert 'localStorage.getItem(\'app-theme\')' in html
    
    # 5. Accessibility ARIA Roles
    assert 'role="tablist"' in html
    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html


def test_frontend_styles_responsive_and_accessibility():
    """Verify CSS has mobile drawer transitions, focus rings, and light mode classes."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css_path = os.path.join(base_dir, "frontend", "styles.css")
    assert os.path.exists(css_path)
    
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    
    # Drawer transitions
    assert '#sidebar.mobile-open' in css
    assert '#sidebar-backdrop.active' in css
    
    # Accessibility focus rings
    assert 'button:focus-visible' in css
    
    # Light mode rules
    assert 'html.light' in css
    assert 'html.light #mobile-bottom-nav' in css
    assert 'html.light #model-modal' in css


def test_frontend_js_router_and_auto_eda_resilience():
    """Verify app.js includes router state persistence, cancellation, and rehydration."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    js_path = os.path.join(base_dir, "frontend", "app.js")
    assert os.path.exists(js_path)
    
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()
    
    # Per-hub navigation state memory
    assert 'lastActiveSubpage' in js
    assert 'handleHashRouting' in js
    
    # Concurrency lock & AbortController cancellation
    assert 'isAnalyzing' in js
    assert 'autoEdaAbortController' in js
    assert 'cancelAutoEDA' in js
    
    # Session rehydration
    assert 'initSessionRehydration' in js
    assert 'eda_session_id' in js
    
    # Model modal search & filters
    assert 'modelTierFilter' in js
    assert 'modelSearchQuery' in js
    assert 'resetModelFilters' in js
