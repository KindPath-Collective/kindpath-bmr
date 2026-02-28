"""
BMR — Market Curvature Index (MCI)
====================================
Measures the tokenisation gap — the distance between
price (the token) and fundamental/intrinsic value (the referent).

  K = (Price − Value) / Value

High positive K: overextension (token > referent)
  → Steep curvature, reversal pressure loading
  → IN-Loading at the asset level

High negative K: underextension (referent > token)
  → Syntropic loading — energy available to release upward
  → ZPB potential — the asset holds more than it's expressing

K near zero: coherence (token ≈ referent)
  → Shallow curvature, ZPB of valuation

Value computation varies by asset class:
  EQUITY    — P/E relative to historical + sector mean
  FOREX     — Purchasing Power Parity deviation
  COMMODITY — Supply/demand equilibrium proxy (price vs 200d MA)
  CRYPTO    — Network value / transaction volume (NVT ratio)
  INDEX     — Cyclically adjusted P/E (CAPE proxy)

Evidence levels:
  Equity valuation:   [ESTABLISHED]
  PPP for forex:      [TESTABLE] — PPP is a long-run model
  Commodity model:    [TESTABLE] — 200d MA proxy is directional only
  Crypto NVT:         [TESTABLE] — NVT as valuation measure is developing
"""

from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, List
from feeds.feeds import OHLCV

logger = logging.getLogger(__name__)

# Asset class detection from symbol
EQUITY_SYMBOLS   = {"SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "GOOGL",
                    "AMZN", "NVDA", "TSLA", "META"}
FOREX_PAIRS      = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
                    "NZDUSD", "USDCHF", "EUR/USD", "GBP/USD"}
COMMODITY_SYMS   = {"GC=F", "CL=F", "SI=F", "NG=F", "HG=F", "ZC=F",
                    "GLD", "USO", "SLV"}
CRYPTO_SYMS      = {"BTC-USD", "ETH-USD", "BTC/USD", "ETH/USD",
                    "BTCUSDT", "ETHUSDT"}
INDEX_SYMS       = {"^GSPC", "^NDX", "^DJI", "^RUT", "^FTSE",
                    "^N225", "^HSI"}


@dataclass
class CurvatureResult:
    """Market curvature reading for one instrument."""
    symbol: str
    asset_class: str
    k: float              # curvature: -1 (underextended) → +1 (overextended)
    price: float
    value_estimate: float
    curvature_state: str  # SYNTROPIC | COHERENT | LOADED | OVEREXTENDED
    evidence_level: str
    notes: str = ""
    method: str = ""


def detect_asset_class(symbol: str) -> str:
    """Detect asset class from symbol."""
    s = symbol.upper()
    if s in CRYPTO_SYMS or "BTC" in s or "ETH" in s:
        return "CRYPTO"
    if s in INDEX_SYMS or s.startswith("^"):
        return "INDEX"
    if s in FOREX_PAIRS or "/" in s:
        return "FOREX"
    if s in COMMODITY_SYMS or s.endswith("=F"):
        return "COMMODITY"
    return "EQUITY"


def compute_curvature(symbol: str,
                      bars: List[OHLCV],
                      extra: Optional[dict] = None) -> CurvatureResult:
    """
    Compute Market Curvature Index for an instrument.

    extra: optional dict with additional data:
      equity:    {"pe_ratio": float, "sector_pe": float}
      forex:     {"ppp_rate": float}
      crypto:    {"nvt_ratio": float}
    """
    if not bars:
        return CurvatureResult(
            symbol=symbol, asset_class="UNKNOWN",
            k=0.0, price=0.0, value_estimate=0.0,
            curvature_state="COHERENT",
            evidence_level="SPECULATIVE",
            notes="No price data"
        )

    asset_class = detect_asset_class(symbol)
    current_price = bars[-1].close

    if asset_class == "EQUITY":
        return _equity_curvature(symbol, bars, current_price, extra or {})
    elif asset_class == "FOREX":
        return _forex_curvature(symbol, bars, current_price, extra or {})
    elif asset_class in ("COMMODITY", "INDEX"):
        return _ma_curvature(symbol, bars, current_price, asset_class)
    elif asset_class == "CRYPTO":
        return _crypto_curvature(symbol, bars, current_price, extra or {})
    else:
        return _ma_curvature(symbol, bars, current_price, "EQUITY")


def _equity_curvature(symbol, bars, price, extra) -> CurvatureResult:
    """
    Equity: P/E relative to historical + sector mean.
    [ESTABLISHED]
    """
    pe = extra.get("pe_ratio")
    sector_pe = extra.get("sector_pe")

    if pe is None or sector_pe is None:
        # Fallback: price vs 200d MA as momentum-of-value proxy
        return _ma_curvature(symbol, bars, price, "EQUITY")

    # K = (P/E - SectorMeanPE) / SectorMeanPE
    k_raw = (pe - sector_pe) / (sector_pe + 1e-10)
    k = float(np.clip(k_raw, -1.0, 1.0))
    value_estimate = price / (1 + k_raw) if k_raw > -1 else price

    return CurvatureResult(
        symbol=symbol, asset_class="EQUITY",
        k=k, price=price, value_estimate=value_estimate,
        curvature_state=_state(k),
        evidence_level="ESTABLISHED",
        notes=f"P/E {pe:.1f} vs sector {sector_pe:.1f}",
        method="pe_relative"
    )


def _forex_curvature(symbol, bars, price, extra) -> CurvatureResult:
    """
    Forex: PPP deviation.
    [TESTABLE] — PPP is a long-run equilibrium model.
    """
    ppp_rate = extra.get("ppp_rate")

    if ppp_rate is None:
        # No PPP data — use 200d MA proxy
        return _ma_curvature(symbol, bars, price, "FOREX")

    k_raw = (price - ppp_rate) / (ppp_rate + 1e-10)
    k = float(np.clip(k_raw * 2, -1.0, 1.0))  # scale: forex PPP gaps tend to be small

    return CurvatureResult(
        symbol=symbol, asset_class="FOREX",
        k=k, price=price, value_estimate=ppp_rate,
        curvature_state=_state(k),
        evidence_level="TESTABLE",
        notes=f"PPP rate {ppp_rate:.4f} vs current {price:.4f}",
        method="ppp_deviation"
    )


def _ma_curvature(symbol, bars, price, asset_class) -> CurvatureResult:
    """
    200-day MA as value proxy.
    [TESTABLE] — directional only, not absolute valuation.
    Price/200d-MA deviation as curvature.
    Historically useful for mean-reversion and trend detection.
    """
    closes = np.array([b.close for b in bars])
    period = min(200, len(closes))
    ma200 = float(np.mean(closes[-period:]))

    k_raw = (price - ma200) / (ma200 + 1e-10)
    # Scale: 20% deviation = K ≈ 0.5 (moderately loaded)
    k = float(np.clip(k_raw / 0.4, -1.0, 1.0))

    return CurvatureResult(
        symbol=symbol, asset_class=asset_class,
        k=k, price=price, value_estimate=ma200,
        curvature_state=_state(k),
        evidence_level="TESTABLE",
        notes=f"Price {price:.4f} vs {period}d MA {ma200:.4f} ({k_raw*100:+.1f}%)",
        method=f"ma{period}_deviation"
    )


def _crypto_curvature(symbol, bars, price, extra) -> CurvatureResult:
    """
    Crypto: NVT ratio proxy (Network Value to Transactions).
    High NVT = high market cap relative to transaction volume = overvalued.
    [TESTABLE] — NVT as crypto valuation metric is developing.

    Without on-chain data, falls back to MA curvature.
    """
    nvt = extra.get("nvt_ratio")

    if nvt is None:
        result = _ma_curvature(symbol, bars, price, "CRYPTO")
        result.evidence_level = "TESTABLE"
        result.notes += " (NVT unavailable — MA proxy used)"
        return result

    # NVT: >150 = overvalued, <25 = undervalued, ~75 = fair
    k_raw = (nvt - 75) / 75
    k = float(np.clip(k_raw, -1.0, 1.0))

    return CurvatureResult(
        symbol=symbol, asset_class="CRYPTO",
        k=k, price=price, value_estimate=price / (1 + k_raw),
        curvature_state=_state(k),
        evidence_level="TESTABLE",
        notes=f"NVT {nvt:.1f} (fair≈75)",
        method="nvt_ratio"
    )


def _state(k: float) -> str:
    """Map K value to curvature state."""
    if k < -0.35:
        return "SYNTROPIC"      # underextended — energy available to release upward
    elif k < 0.15:
        return "COHERENT"       # price ≈ value
    elif k < 0.50:
        return "LOADED"         # building overextension
    else:
        return "OVEREXTENDED"   # steep curvature — reversal pressure
