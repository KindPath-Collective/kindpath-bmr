"""
BMR — Signal Server
====================
FastAPI server exposing the full BMR pipeline as a REST API.
Same architecture as q_server.py in KindPath Q.

Endpoints:
  GET  /              — service info
  GET  /ping          — connectivity check
  GET  /status        — capability report
  POST /analyse       — full BMR analysis for a symbol
  POST /analyse/multi — multi-symbol sweep (basket analysis)
  GET  /api/nu_score  — text-domain coherence ν score (tab_scout integration)

Usage:
  export FRED_API_KEY=your_key_here
  python bmr_server.py

  Then from DFTE or FIELD app:
  POST http://localhost:8001/analyse
  {"symbol": "SPY", "timeframe": "1d"}
"""

from __future__ import annotations
import os
import sys
import time
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feeds.feeds import (
    OHLCVFeed, MomentumSignal, VolumePressureSignal, OptionsSkewSignal,
    COTSignal, InstitutionalFlowSignal, CreditSpreadSignal,
    MacroSignal, CentralBankSignal, GeopoliticalSignal,
    FREDMetaSignal,
)
from core.normaliser import normalise_scale
from core.nu_engine import compute_nu, compute_multi_timeframe_nu
from core.lsii_price import compute_lsii_price
from core.curvature import compute_curvature
from core.bmr_profile import synthesise_bmr_profile, BMRProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bmr_server")

app = FastAPI(
    title="BMR Signal Server",
    description="Behavioural Market Relativity — KindPath Trading Engine Signal Layer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVER_VERSION = "1.0.0"


# ─── Request / Response models ────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    periods: int = 200
    include_lsii: bool = True
    include_curvature: bool = True
    multi_timeframe: bool = False
    extra: dict = {}


class MultiAnalyseRequest(BaseModel):
    symbols: List[str]
    timeframe: str = "1d"
    periods: int = 200


# ─── Profile serialiser ───────────────────────────────────────────────────────

def _serialise_profile(profile: BMRProfile) -> dict:
    """Convert BMRProfile to JSON-safe dict."""
    def _safe(v):
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    return {
        "symbol":         profile.symbol,
        "timestamp":      profile.timestamp.isoformat(),
        "mfs":            _safe(profile.mfs),
        "mfs_label":      profile.mfs_label,
        "direction":      _safe(profile.direction),
        "nu": {
            "score":       _safe(profile.nu),
            "field_state": profile.field_state,
            "scales":      {k: _safe(v) for k, v in profile.scale_values.items()},
        },
        "lsii": {
            "score":       _safe(profile.lsii),
            "flag":        profile.lsii_flag,
            "late_break":  profile.late_move_break,
        } if profile.lsii is not None else None,
        "curvature": {
            "k":             _safe(profile.k),
            "state":         profile.curvature_state,
            "value_estimate": _safe(profile.value_estimate),
        } if profile.k is not None else None,
        "trade_tier":      profile.trade_tier,
        "tier_rationale":  profile.tier_rationale,
        "interpretation":  profile.interpretation,
        "field_note":      profile.field_note,
        "recommendations": profile.recommendations,
        "components": [
            {
                "name":           c.name,
                "score":          _safe(c.score),
                "weight":         _safe(c.weight),
                "evidence_level": c.evidence_level,
                "notes":          c.notes,
            }
            for c in profile.components
        ],
        "evidence_notes": profile.evidence_notes,
        "_meta": {
            "server_version": SERVER_VERSION,
            "fred_available": bool(os.environ.get("FRED_API_KEY")),
        },
    }


# ─── Core pipeline ────────────────────────────────────────────────────────────

def _run_pipeline(req: AnalyseRequest) -> dict:
    """Full BMR pipeline for one symbol."""
    t0 = time.time()
    symbol = req.symbol
    tf = req.timeframe

    # 1. Fetch OHLCV
    feed = OHLCVFeed(source="yahoo")
    bars = feed.fetch(symbol, tf, req.periods)
    if not bars:
        raise HTTPException(400, f"No price data for {symbol}")

    # 2. Participant signals
    p_signals = [
        MomentumSignal().compute(bars, symbol),
        VolumePressureSignal().compute(bars, symbol),
        OptionsSkewSignal().compute(symbol),
    ]

    # 3. Institutional signals
    i_signals = [
        COTSignal().compute(symbol),
        InstitutionalFlowSignal().compute(bars, symbol),
        CreditSpreadSignal().compute(symbol),
    ]

    # 4. Sovereign signals
    s_signals = [
        MacroSignal().compute(symbol),
        CentralBankSignal().compute(symbol),
        GeopoliticalSignal().compute(symbol),
    ]

    # 5. Normalise scales
    p_reading = normalise_scale(p_signals, "PARTICIPANT")
    i_reading = normalise_scale(i_signals, "INSTITUTIONAL")
    s_reading = normalise_scale(s_signals, "SOVEREIGN")

    # 6. Compute ν
    nu_result = compute_nu(p_reading, i_reading, s_reading)

    # 7. LSII-Price
    lsii_result = None
    if req.include_lsii and len(bars) >= 20:
        lsii_result = compute_lsii_price(bars)

    # 8. Market curvature
    curvature_result = None
    if req.include_curvature:
        curvature_result = compute_curvature(symbol, bars, req.extra)

    # 9. Multi-timeframe ν
    multi_tf = None
    if req.multi_timeframe:
        try:
            tf_map = {
                "macro":    ("1mo", 48),
                "swing":    ("1wk", 52),
                "intraday": ("1d",  200),
            }
            tf_readings = {}
            for tf_name, (tf_code, periods) in tf_map.items():
                tf_bars = feed.fetch(symbol, tf_code, periods)
                if len(tf_bars) >= 20:
                    tf_p = normalise_scale([
                        MomentumSignal().compute(tf_bars, symbol),
                        VolumePressureSignal().compute(tf_bars, symbol),
                    ], "PARTICIPANT")
                    tf_i = normalise_scale([
                        InstitutionalFlowSignal().compute(tf_bars, symbol),
                    ], "INSTITUTIONAL")
                    tf_s = normalise_scale([
                        CentralBankSignal().compute(symbol),
                    ], "SOVEREIGN")
                    tf_readings[tf_name] = (tf_p, tf_i, tf_s)
            if tf_readings:
                multi_tf = compute_multi_timeframe_nu(tf_readings)
        except Exception as e:
            logger.warning(f"Multi-TF failed for {symbol}: {e}")

    # 10. Synthesise profile
    profile = synthesise_bmr_profile(
        symbol=symbol,
        nu_result=nu_result,
        lsii_result=lsii_result,
        curvature_result=curvature_result,
        multi_tf=multi_tf,
    )

    # 11. Extended FRED metadata (Sovereign Context)
    fred_meta = {}
    try:
        fred_meta = FREDMetaSignal().compute()
    except Exception as e:
        logger.warning(f"FREDMeta computation failed: {e}")

    result = _serialise_profile(profile)
    result["_meta"]["fred_meta"] = fred_meta
    result["_meta"]["elapsed_s"] = round(time.time() - t0, 2)
    return result


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name":    "BMR Signal Server",
        "version": SERVER_VERSION,
        "equation": "M = [(Participant × Institutional × Sovereign) · ν]²",
        "endpoints": ["/ping", "/status", "/analyse", "/analyse/multi", "/api/nu_score"],
    }


@app.get("/ping")
async def ping():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/status")
async def status():
    caps = {
        "yfinance_available": False,
        "fred_available":     bool(os.environ.get("FRED_API_KEY")),
        "cot_available":      True,
        "server_version":     SERVER_VERSION,
    }
    try:
        import yfinance
        caps["yfinance_available"] = True
    except ImportError:
        pass
    return caps


@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    """
    Full BMR analysis for a single symbol.
    Returns complete BMRProfile as JSON.
    """
    try:
        return _run_pipeline(req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {req.symbol}: {traceback.format_exc()}")
        raise HTTPException(500, f"Analysis error: {str(e)}")


@app.get("/api/nu_score")
async def nu_score_text(text: str = ""):
    """
    Text-domain coherence ν score for KindPath relevance.

    Interprets ν as cross-domain coherence: how many KindPath domains does
    this text simultaneously engage? Multi-domain signal = high ν, mirroring
    the financial ν where P × I × S coherence amplifies the field.

    Used by kindai/missions/tab_scout.py to weight page relevance against the
    KindPath knowledge base before ingestion.

    Returns {"nu": float, "domains_active": list, "matched_terms": list}
    """
    import math

    # Domain vocabulary — each domain is a conceptual "scale" in the ν model
    _DOMAIN_VOCAB: dict[str, list[str]] = {
        "ndis_care": [
            "ndis", "participant", "support worker", "disability", "care plan",
            "allied health", "occupational therapy", "sbas", "plan management",
            "capacity building", "supported independent living", "sil", "ndis provider",
            "psychosocial", "recovery coach", "plan review", "early childhood",
        ],
        "trading_finance": [
            "dfte", "trading", "kepe", "market", "momentum", "options", "futures",
            "portfolio", "alpha", "signal", "strategy", "backtest", "hedge",
            "bitcoin", "crypto", "equity", "forex", "yield", "inflation",
            "interest rate", "bull", "bear", "liquidity", "volatility",
        ],
        "kindpath_mission": [
            "kindpath", "kindfluence", "kindearth", "kindfield", "kindsense",
            "bmr", "benevolence", "syntropy", "sovereignty", "collective",
            "field", "compass", "regenerative", "relational", "lsii",
        ],
        "music_audio": [
            "psychosomatic", "frequency", "spectral", "harmonic", "vocal",
            "fingerprint", "seedbank", "juce", "daw", "plugin", "vst", "au",
            "compression", "mix", "master", "producer", "audio",
            "music", "sound design", "ableton", "logic pro",
        ],
        "tech_ai": [
            "python", "fastapi", "llm", "artificial intelligence", "machine learning",
            "cloud", "docker", "microservice", "model", "embedding", "vector",
            "prompt", "agent", "api", "github", "copilot", "automation",
        ],
        "study_opportunity": [
            "research", "study", "paper", "journal", "university", "course",
            "certification", "learning", "education", "tutorial", "documentation",
            "grant", "funding", "tender", "proposal", "opportunity", "scholarship",
        ],
    }

    if not text:
        return {"nu": 0.0, "domains_active": [], "matched_terms": []}

    lowered = text.lower()

    domain_scores: dict[str, float] = {}
    all_matched: list[str] = []

    for domain, terms in _DOMAIN_VOCAB.items():
        hits = [t for t in terms if t in lowered]
        if hits:
            all_matched.extend(hits)
            # Weight by term specificity (longer/multi-word phrases score higher)
            score = sum(1.0 + math.log1p(len(t.split())) for t in hits)
            max_score = sum(1.0 + math.log1p(len(t.split())) for t in terms)
            domain_scores[domain] = min(score / max_score, 1.0) if max_score else 0.0

    domains_active = [d for d, s in domain_scores.items() if s > 0.05]

    if not domains_active:
        return {"nu": 0.0, "domains_active": [], "matched_terms": []}

    # ν = cross-domain coherence (mirrors P × I × S model)
    # Single domain caps at ~0.4; three+ domains pushes toward 1.0
    n = len(domains_active)
    domain_strength = sum(domain_scores[d] for d in domains_active) / n
    coherence_factor = 1.0 - math.exp(-0.8 * n)  # asymptotic approach to 1.0

    nu = round(domain_strength * coherence_factor, 3)
    nu = min(max(nu, 0.0), 1.0)

    return {
        "nu":            nu,
        "domains_active": sorted(domains_active),
        "matched_terms": sorted(set(all_matched))[:20],
    }


@app.post("/analyse/multi")
async def analyse_multi(req: MultiAnalyseRequest):
    """
    Analyse a basket of symbols and return MFS + ν for each.
    Useful for sector coherence mapping and KEPE integration.
    """
    results = {}
    errors = {}
    for symbol in req.symbols:
        try:
            single_req = AnalyseRequest(
                symbol=symbol,
                timeframe=req.timeframe,
                periods=req.periods,
                include_lsii=False,      # speed optimisation for basket
                include_curvature=True,
                multi_timeframe=False,
            )
            results[symbol] = _run_pipeline(single_req)
        except Exception as e:
            errors[symbol] = str(e)
            logger.warning(f"Basket analysis failed for {symbol}: {e}")

    # Basket-level coherence: average ν across all symbols
    nu_vals = [r["nu"]["score"] for r in results.values() if "nu" in r]
    basket_nu = float(np.mean(nu_vals)) if nu_vals else 0.0

    return {
        "basket_nu":     basket_nu,
        "symbol_count":  len(req.symbols),
        "success_count": len(results),
        "results":       results,
        "errors":        errors,
        "_meta":         {"server_version": SERVER_VERSION},
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("BMR_PORT", 8001))
    logger.info(f"BMR Signal Server starting on port {port}")
    logger.info(f"FRED API key: {'set' if os.environ.get('FRED_API_KEY') else 'NOT SET'}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
