"""
BMR — ν (Nu) Engine
====================
Computes the cross-scale frequency coherence coefficient.

ν measures whether Participant, Institutional, and Sovereign
scales are singing the same song.

  ν = 1.0  →  All scales fully aligned (Coherent Trend)
  ν = 0.5  →  Partial alignment (Transition/DRIFT)
  ν = 0.2  →  Scales diverging (IN-Loading / Compression)
  ν < 0.15 →  Coherence collapse (SIC Event)

The ν coefficient gates the M² amplification:
  M = [(P × I × S) · ν]²

High ν amplifies directional energy.
Low ν dissipates it into noise.

This is not a sentiment indicator.
It is a coherence indicator.
The distinction is the contribution.
"""

from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from core.normaliser import ScaleReading

logger = logging.getLogger(__name__)

# Timeframe hierarchy weights for multi-timeframe ν
TIMEFRAME_WEIGHTS = {
    "macro":    0.40,  # monthly/quarterly — sovereign field container
    "swing":    0.35,  # daily/weekly — institutional rhythm
    "intraday": 0.25,  # session — participant flow
}

# ν thresholds → Field State
NU_THRESHOLDS = {
    "ZPB":        0.75,   # Coherent Trend
    "DRIFT":      0.40,   # Transition
    "IN_LOADING": 0.15,   # Compression — suppressed energy
    "SIC":        0.00,   # Coherence collapse
}


@dataclass
class NuResult:
    """
    The ν coherence reading for a given instrument/timeframe combination.
    """
    nu: float                          # 0.0 → 1.0
    field_state: str                   # ZPB | DRIFT | IN_LOADING | SIC
    direction: float                   # -1.0 → +1.0 (net directional signal)
    scale_values: Dict[str, float]     # per-scale directional values
    scale_confidences: Dict[str, float]
    pairwise_coherence: Dict[str, float]  # P-I, P-S, I-S divergence
    amplified_m: float                 # M = [(P × I × S) · ν]²
    notes: str = ""

    @property
    def is_tradeable(self) -> bool:
        """Field is coherent enough to trend-follow."""
        return self.nu >= NU_THRESHOLDS["DRIFT"]

    @property
    def is_compressed(self) -> bool:
        """IN-Loading: suppressed energy, volatility expansion likely."""
        return NU_THRESHOLDS["IN_LOADING"] <= self.nu < NU_THRESHOLDS["DRIFT"]

    @property
    def is_sic_event(self) -> bool:
        """Coherence collapse — SIC event in progress."""
        return self.nu < NU_THRESHOLDS["IN_LOADING"]


@dataclass
class MultiTimeframeNu:
    """
    ν computed across multiple timeframes.
    Each timeframe has its own ScaleReadings and NuResult.
    Total ν is the weighted product — all timeframes must agree
    for maximum field energy.
    """
    timeframe_results: Dict[str, NuResult]
    total_nu: float
    total_field_state: str
    total_direction: float
    dominant_timeframe: str


def compute_nu(
    participant: ScaleReading,
    institutional: ScaleReading,
    sovereign: ScaleReading,
) -> NuResult:
    """
    Compute ν from three scale readings.

    ν = 1 - mean_pairwise_divergence

    Divergence between two scales:
      div(A, B) = |value_A - value_B| / 2
      (normalised to 0→1 since values are -1→+1)

    Weighted by confidence of weaker reading in each pair.
    """
    p_val = participant.value
    i_val = institutional.value
    s_val = sovereign.value

    p_conf = participant.confidence
    i_conf = institutional.confidence
    s_conf = sovereign.confidence

    # Pairwise divergences (0 = identical, 1 = opposite)
    div_pi = abs(p_val - i_val) / 2.0
    div_ps = abs(p_val - s_val) / 2.0
    div_is = abs(i_val - s_val) / 2.0

    # Weight each pair by the confidence of the weaker signal
    # (a divergence between two confident signals is more meaningful
    # than a divergence where one signal is low confidence)
    w_pi = min(p_conf, i_conf)
    w_ps = min(p_conf, s_conf)
    w_is = min(i_conf, s_conf)

    total_w = w_pi + w_ps + w_is + 1e-10
    mean_div = (div_pi * w_pi + div_ps * w_ps + div_is * w_is) / total_w

    nu = float(np.clip(1.0 - mean_div, 0.0, 1.0))

    # Determine field state
    field_state = "SIC"
    for state, threshold in sorted(NU_THRESHOLDS.items(), key=lambda x: -x[1]):
        if nu >= threshold:
            field_state = state
            break

    # Net directional signal — confidence-weighted mean of scale values
    total_conf = p_conf + i_conf + s_conf + 1e-10
    direction = float(np.clip(
        (p_val * p_conf + i_val * i_conf + s_val * s_conf) / total_conf,
        -1.0, 1.0
    ))

    # M = [(P × I × S) · ν]²
    # Scale product: average of absolute values × sign of direction
    scale_product = (abs(p_val) * abs(i_val) * abs(s_val)) ** (1/3)  # geometric mean
    m_raw = scale_product * nu
    amplified_m = float(np.clip(m_raw ** 2, 0.0, 1.0)) * np.sign(direction)

    # Build notes
    notes_parts = []
    if div_pi > 0.5:
        notes_parts.append(f"P-I divergence: {div_pi:.3f} (participant vs institutional split)")
    if div_ps > 0.5:
        notes_parts.append(f"P-S divergence: {div_ps:.3f} (participant vs sovereign split)")
    if div_is > 0.5:
        notes_parts.append(f"I-S divergence: {div_is:.3f} (institutional vs sovereign split)")
    if not notes_parts:
        notes_parts.append(f"Scales coherent — ν {nu:.3f}")

    return NuResult(
        nu=nu,
        field_state=field_state,
        direction=direction,
        scale_values={
            "participant":   p_val,
            "institutional": i_val,
            "sovereign":     s_val,
        },
        scale_confidences={
            "participant":   p_conf,
            "institutional": i_conf,
            "sovereign":     s_conf,
        },
        pairwise_coherence={
            "participant_institutional": 1 - div_pi,
            "participant_sovereign":     1 - div_ps,
            "institutional_sovereign":   1 - div_is,
        },
        amplified_m=float(amplified_m),
        notes=" | ".join(notes_parts),
    )


def compute_multi_timeframe_nu(
    timeframe_readings: Dict[str, Tuple[ScaleReading, ScaleReading, ScaleReading]]
) -> MultiTimeframeNu:
    """
    Compute ν across multiple timeframes and derive total field coherence.

    timeframe_readings: {
      "macro":    (participant, institutional, sovereign),
      "swing":    (participant, institutional, sovereign),
      "intraday": (participant, institutional, sovereign),
    }

    Total ν = weighted product of timeframe ν values.
    Macro carries highest weight — sovereign field sets the container.
    """
    timeframe_results = {}
    for tf, (p, i, s) in timeframe_readings.items():
        timeframe_results[tf] = compute_nu(p, i, s)

    # Weighted product for total ν
    # Product because ALL timeframes must agree for maximum energy
    # (one mis-aligned timeframe significantly reduces total)
    weighted_log_nu = 0.0
    weight_total = 0.0
    for tf, result in timeframe_results.items():
        w = TIMEFRAME_WEIGHTS.get(tf, 0.25)
        # Log-space weighted mean (product equivalent)
        nu_safe = max(result.nu, 0.001)  # avoid log(0)
        weighted_log_nu += np.log(nu_safe) * w
        weight_total += w

    total_nu = float(np.clip(np.exp(weighted_log_nu / weight_total), 0.0, 1.0))

    # Total field state
    total_field_state = "SIC"
    for state, threshold in sorted(NU_THRESHOLDS.items(), key=lambda x: -x[1]):
        if total_nu >= threshold:
            total_field_state = state
            break

    # Direction: macro-weighted
    macro_result = timeframe_results.get("macro")
    swing_result = timeframe_results.get("swing")
    intraday_result = timeframe_results.get("intraday")

    direction = 0.0
    if macro_result:
        direction += macro_result.direction * 0.50
    if swing_result:
        direction += swing_result.direction * 0.30
    if intraday_result:
        direction += intraday_result.direction * 0.20
    direction = float(np.clip(direction, -1.0, 1.0))

    # Dominant timeframe = highest ν (most coherent)
    if timeframe_results:
        dominant_timeframe = max(timeframe_results, key=lambda tf: timeframe_results[tf].nu)
    else:
        dominant_timeframe = "none"

    return MultiTimeframeNu(
        timeframe_results=timeframe_results,
        total_nu=total_nu,
        total_field_state=total_field_state,
        total_direction=direction,
        dominant_timeframe=dominant_timeframe,
    )
