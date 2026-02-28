"""
BMR — LSII-Price (Late-Move Inversion Index)
=============================================
Direct translation of KindPath Q's LSII divergence engine
into the price domain.

Original hypothesis (music):
  Authentic creative expression sometimes breaks the established arc
  in Q4 — the creator's genuine voice asserting itself.

Market translation:
  Authentic structural conviction continues its arc in Q4.
  Distribution / exhaustion / institutional withdrawal
  breaks the arc in Q4 — the move's genuine character
  asserting itself against the surface narrative.

Q4 divergence from Q1–Q3 baseline across four axes:
  MOMENTUM  → RSI / rate-of-change trajectory
  DYNAMIC   → Volume character relative to baseline
  TIMBRAL   → Volatility (ATR) trajectory
  TEMPORAL  → Candle body proportion (conviction per candle)

High LSII-Price = Q4 breaking its own arc
  → Distribution in uptrend (institutional withdrawal)
  → Capitulation exhaustion in downtrend
  → Structural integrity failure

Low LSII-Price = Q4 confirms Q1–Q3
  → Move has internal coherence
  → Continuation more likely than reversal

Evidence level: [TESTABLE]
Requires backtesting validation against instrument corpus.
"""

from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from feeds.feeds import OHLCV

logger = logging.getLogger(__name__)

# LSII-Price thresholds (inherited from KindPath Q, to be calibrated)
LSII_THRESHOLDS = {
    "none":      0.05,
    "low":       0.12,
    "moderate":  0.22,
    "high":      0.35,
    "very_high": 0.50,
}


@dataclass
class PriceQuarter:
    """One temporal quarter of a price move."""
    label: str          # Q1, Q2, Q3, Q4
    bars: List[OHLCV]
    momentum: float     # mean RSI-equivalent
    volume:   float     # mean volume relative to full-move baseline
    volatility: float   # mean ATR relative to full-move baseline
    conviction: float   # mean candle body proportion (close-open / high-low)


@dataclass
class LsiiPriceResult:
    """LSII-Price result for a price structure."""
    lsii: float
    flag_level: str       # none | low | moderate | high | very_high
    direction: str        # stable | expanding | contracting | inverting
    dominant_axis: str    # momentum | dynamic | timbral | temporal
    description: str
    flag_notes: str
    evidence_level: str = "TESTABLE"

    # Per-axis divergence
    momentum_divergence:   float = 0.0
    dynamic_divergence:    float = 0.0
    volatility_divergence: float = 0.0
    conviction_divergence: float = 0.0

    # Quarter values for visualisation
    momentum_arc:   List[float] = field(default_factory=list)
    volume_arc:     List[float] = field(default_factory=list)
    volatility_arc: List[float] = field(default_factory=list)
    conviction_arc: List[float] = field(default_factory=list)


def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Rolling RSI."""
    rsi = np.full(len(closes), 50.0)
    if len(closes) < period + 1:
        return rsi
    deltas = np.diff(closes)
    for i in range(period, len(closes)):
        gains = np.where(deltas[i-period:i] > 0, deltas[i-period:i], 0)
        losses = np.where(deltas[i-period:i] < 0, -deltas[i-period:i], 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    return rsi


def _compute_atr(bars: List[OHLCV], period: int = 14) -> np.ndarray:
    """Average True Range."""
    if len(bars) < 2:
        return np.zeros(len(bars))
    tr = []
    for i in range(1, len(bars)):
        hl = bars[i].high - bars[i].low
        hc = abs(bars[i].high - bars[i-1].close)
        lc = abs(bars[i].low - bars[i-1].close)
        tr.append(max(hl, hc, lc))
    tr = np.array([tr[0]] + tr)
    atr = np.zeros(len(bars))
    for i in range(len(bars)):
        start = max(0, i - period + 1)
        atr[i] = np.mean(tr[start:i+1])
    return atr


def _conviction(bar: OHLCV) -> float:
    """Candle body proportion — how much of the range was directional movement."""
    hl = bar.high - bar.low
    if hl < 1e-10:
        return 0.5
    body = abs(bar.close - bar.open)
    return float(body / hl)


def _divergence(baseline: List[float], q4_value: float) -> float:
    """
    How much does q4_value diverge from the Q1–Q3 baseline trend?
    Identical to KindPath Q's divergence function.
    """
    if not baseline:
        return 0.0
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline) + 1e-10
    return abs(q4_value - baseline_mean) / (baseline_std * 3 + 1e-10)


def _quarter_features(bars: List[OHLCV], baseline_vol: float,
                      baseline_atr: float) -> PriceQuarter:
    """Extract axis values from a quarter of bars."""
    if not bars:
        return PriceQuarter("?", [], 50.0, 1.0, 1.0, 0.5)

    closes = np.array([b.close for b in bars])
    rsi_vals = _compute_rsi(closes)
    atr_vals = _compute_atr(bars)

    momentum = float(np.mean(rsi_vals))
    vol_mean = float(np.mean([b.volume for b in bars]))
    volume = vol_mean / (baseline_vol + 1e-10)
    atr_mean = float(np.mean(atr_vals))
    volatility = atr_mean / (baseline_atr + 1e-10)
    conviction = float(np.mean([_conviction(b) for b in bars]))

    return PriceQuarter(
        label="Qn", bars=bars,
        momentum=momentum, volume=volume,
        volatility=volatility, conviction=conviction,
    )


def compute_lsii_price(bars: List[OHLCV],
                       min_bars: int = 20) -> LsiiPriceResult:
    """
    Compute LSII-Price over a sequence of bars representing one price move.

    The bars should cover one complete structural leg:
      - A trend move (e.g. impulse from swing low to swing high)
      - A session (e.g. full trading day)
      - A defined period (e.g. weekly bar sequence)

    Caller is responsible for identifying the structural boundary.
    Minimum 20 bars for reliable computation.
    """
    if len(bars) < min_bars:
        return LsiiPriceResult(
            lsii=0.0, flag_level="none",
            direction="stable", dominant_axis="none",
            description=f"Insufficient bars ({len(bars)} < {min_bars})",
            flag_notes="Need more price data",
        )

    n = len(bars)
    q_len = n // 4

    # Baseline stats for normalisation
    baseline_vol = float(np.mean([b.volume for b in bars])) + 1e-10
    baseline_atr = float(np.mean(_compute_atr(bars))) + 1e-10

    # Divide into quarters
    quarters = []
    labels = ["Q1", "Q2", "Q3", "Q4"]
    for i, label in enumerate(labels):
        start = i * q_len
        end = (i + 1) * q_len if i < 3 else n
        qf = _quarter_features(bars[start:end], baseline_vol, baseline_atr)
        qf.label = label
        quarters.append(qf)

    # Extract per-axis Q1–Q3 baselines and Q4 values
    mom_vals  = [q.momentum   for q in quarters]
    vol_vals  = [q.volume     for q in quarters]
    atr_vals  = [q.volatility for q in quarters]
    conv_vals = [q.conviction for q in quarters]

    mom_baseline  = mom_vals[:3]
    vol_baseline  = vol_vals[:3]
    atr_baseline  = atr_vals[:3]
    conv_baseline = conv_vals[:3]

    # Per-axis divergence
    mom_div  = _divergence(mom_baseline,  mom_vals[3])
    dyn_div  = _divergence(vol_baseline,  vol_vals[3])
    atr_div  = _divergence(atr_baseline,  atr_vals[3])
    conv_div = _divergence(conv_baseline, conv_vals[3])

    # Weighted LSII — momentum and conviction weighted slightly higher
    # (most musically analogous to harmonic + timbral in Q)
    lsii = (
        mom_div  * 0.30 +
        dyn_div  * 0.25 +
        atr_div  * 0.25 +
        conv_div * 0.20
    )
    lsii = float(min(lsii, 1.5))

    # Flag level
    flag_level = "none"
    for level in ["very_high", "high", "moderate", "low"]:
        if lsii >= LSII_THRESHOLDS[level]:
            flag_level = level
            break

    # Dominant axis
    axis_scores = {
        "momentum":   mom_div,
        "dynamic":    dyn_div,
        "volatility": atr_div,
        "conviction": conv_div,
    }
    dominant_axis = max(axis_scores, key=axis_scores.get)

    # Direction: compare Q4 to Q3
    q3_mom = mom_vals[2]
    q4_mom = mom_vals[3]
    q3_conv = conv_vals[2]
    q4_conv = conv_vals[3]

    if lsii < LSII_THRESHOLDS["low"]:
        direction = "stable"
    elif q4_mom > q3_mom + 5 or q4_conv > q3_conv + 0.1:
        direction = "expanding"
    elif q4_mom < q3_mom - 5 or q4_conv < q3_conv - 0.1:
        direction = "contracting"
    else:
        direction = "inverting"

    # Description
    mean_q1q3_mom = float(np.mean(mom_baseline))
    trend_char = (
        "High-momentum arc Q1–Q3" if mean_q1q3_mom > 65 else
        "Low-momentum arc Q1–Q3" if mean_q1q3_mom < 35 else
        "Moderate-momentum arc Q1–Q3"
    )

    if flag_level == "none":
        inv = "Q4 confirms arc — move has internal coherence"
    elif flag_level == "low":
        inv = "Slight Q4 deviation — within normal variation"
    elif flag_level == "moderate":
        inv = f"Moderate Q4 divergence on {dominant_axis} — watch for structure break"
    elif flag_level == "high":
        inv = f"Significant Q4 inversion on {dominant_axis} — arc integrity failing"
    else:
        inv = f"Marked Q4 inversion on {dominant_axis} — distribution / exhaustion signal"

    # Flag notes
    notes_parts = []
    if mom_div > 0.3:
        notes_parts.append(f"Momentum divergence: {mom_div:.3f}")
    if dyn_div > 0.3:
        notes_parts.append(f"Volume divergence: {dyn_div:.3f}")
    if atr_div > 0.3:
        notes_parts.append(f"Volatility divergence: {atr_div:.3f}")
    if conv_div > 0.3:
        notes_parts.append(f"Conviction divergence: {conv_div:.3f}")
    flag_notes = " | ".join(notes_parts) if notes_parts else "No significant axis flags"

    return LsiiPriceResult(
        lsii=lsii,
        flag_level=flag_level,
        direction=direction,
        dominant_axis=dominant_axis,
        description=f"{trend_char}. {inv}",
        flag_notes=flag_notes,
        momentum_divergence=mom_div,
        dynamic_divergence=dyn_div,
        volatility_divergence=atr_div,
        conviction_divergence=conv_div,
        momentum_arc=mom_vals,
        volume_arc=vol_vals,
        volatility_arc=atr_vals,
        conviction_arc=conv_vals,
    )
