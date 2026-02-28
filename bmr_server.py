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
from datetime import datetime
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

    result = _serialise_profile(profile)
    result["_meta"]["elapsed_s"] = round(time.time() - t0, 2)
    return result


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name":    "BMR Signal Server",
        "version": SERVER_VERSION,
        "equation": "M = [(Participant × Institutional × Sovereign) · ν]²",
        "endpoints": ["/ping", "/status", "/analyse", "/analyse/multi"],
    }


@app.get("/ping")
async def ping():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


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
