"""
BMR — Profile Synthesiser
===========================
Converges all analytical layers into a single
Market Field Score (MFS) — the BMR field reading of an instrument.

Analogous to KindPath Q's psych_profile.py:
  LSII divergence         → structural integrity signal
  ν coherence             → scale alignment
  Market Curvature        → tokenisation gap
  Scale readings          → directional weight

Output: BMRProfile — the core BMR reading.

This is the integration layer consumed by DFTE.
"""

from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime

from core.nu_engine import NuResult, MultiTimeframeNu, NU_THRESHOLDS
from core.lsii_price import LsiiPriceResult
from core.curvature import CurvatureResult

logger = logging.getLogger(__name__)


@dataclass
class MFSComponent:
    """A scored component contributing to Market Field Score."""
    name: str
    score: float        # 0.0 → 1.0 (directional: 0 = strong bear, 0.5 = neutral, 1 = strong bull)
    weight: float
    evidence_level: str
    source: str
    notes: str = ""


@dataclass
class BMRProfile:
    """
    Full BMR field reading for one instrument.
    The market-side equivalent of KQProfile.
    """
    symbol: str
    timestamp: datetime

    # Core scores
    mfs: float          # Market Field Score 0.0 → 1.0
    mfs_label: str      # ZPB | DRIFT | IN_LOADING | DISRUPTED
    direction: float    # -1.0 → +1.0 net directional bias

    # ν coherence
    nu: float
    field_state: str
    scale_values: Dict[str, float] = field(default_factory=dict)

    # LSII-Price
    lsii: Optional[float] = None
    lsii_flag: Optional[str] = None
    late_move_break: bool = False

    # Curvature
    k: Optional[float] = None
    curvature_state: Optional[str] = None
    value_estimate: Optional[float] = None

    # Components
    components: List[MFSComponent] = field(default_factory=list)

    # Trade tier recommendation
    trade_tier: str = ""    # NANO | MID | LARGE | WAIT
    tier_rationale: str = ""

    # Evidence and interpretation
    interpretation: str = ""
    field_note: str = ""
    evidence_notes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def synthesise_bmr_profile(
    symbol: str,
    nu_result: NuResult,
    lsii_result: Optional[LsiiPriceResult] = None,
    curvature_result: Optional[CurvatureResult] = None,
    multi_tf: Optional[MultiTimeframeNu] = None,
) -> BMRProfile:
    """
    Synthesise all BMR analytical layers into a single profile.
    """
    profile = BMRProfile(
        symbol=symbol,
        timestamp=datetime.utcnow(),
        mfs=0.5,
        mfs_label="DRIFT",
        direction=nu_result.direction,
        nu=nu_result.nu,
        field_state=nu_result.field_state,
        scale_values=nu_result.scale_values,
    )

    components: List[MFSComponent] = []

    # ── LAYER 1: ν Coherence — weight 0.35 ───────────────────────────────────
    # ν as directional amplifier: high ν + direction → strong signal
    # High ν with no direction = coherent ranging (wait)
    nu = nu_result.nu
    direction = nu_result.direction

    nu_score = float(0.5 + (direction * nu * 0.5))  # 0→1 scale
    nu_score = float(np.clip(nu_score, 0.0, 1.0))

    components.append(MFSComponent(
        name="Nu_Coherence",
        score=nu_score,
        weight=0.35,
        evidence_level="TESTABLE",
        source="nu_engine",
        notes=f"ν={nu:.3f} | State={nu_result.field_state} | Dir={direction:+.3f}"
    ))

    # ── LAYER 2: LSII-Price — weight 0.25 ────────────────────────────────────
    if lsii_result:
        profile.lsii = lsii_result.lsii
        profile.lsii_flag = lsii_result.flag_level
        profile.late_move_break = lsii_result.flag_level in ("high", "very_high")

        # High LSII in direction of trade = structural risk (reduce confidence)
        # High LSII against direction = exhaustion of opposing move (opportunity)
        lsii_val = lsii_result.lsii
        lsii_flag_val = {
            "none": 0.0, "low": 0.2, "moderate": 0.4,
            "high": 0.7, "very_high": 1.0
        }.get(lsii_result.flag_level, 0.0)

        # If move is directionally aligned with our signal and has high LSII
        # → structural break in that move → caution
        if direction > 0.2 and lsii_result.direction in ("contracting", "inverting"):
            lsii_score = float(0.5 - lsii_flag_val * 0.3)  # bullish but flagged
        elif direction < -0.2 and lsii_result.direction in ("expanding",):
            lsii_score = float(0.5 + lsii_flag_val * 0.3)  # bearish + bullish exhaustion = short
        else:
            lsii_score = 0.5  # neutral / unclear

        lsii_score = float(np.clip(lsii_score, 0.0, 1.0))

        components.append(MFSComponent(
            name="LSII_Price",
            score=lsii_score,
            weight=0.25,
            evidence_level="TESTABLE",
            source="lsii_price",
            notes=f"LSII={lsii_val:.4f} [{lsii_result.flag_level}] | {lsii_result.direction}"
        ))

    # ── LAYER 3: Market Curvature — weight 0.25 ───────────────────────────────
    if curvature_result:
        profile.k = curvature_result.k
        profile.curvature_state = curvature_result.curvature_state
        profile.value_estimate = curvature_result.value_estimate

        k = curvature_result.k
        # K: -1 (deeply undervalued/syntropic) → +1 (overextended/reversal pressure)
        # For bullish direction: low K = good (room to run), high K = risk
        # curvature_score: high (→1) when setup is favourable, low (→0) when not
        if direction > 0:
            # Bullish: prefer negative/neutral K
            curvature_score = float(np.clip(0.5 - k * 0.4, 0.0, 1.0))
        elif direction < 0:
            # Bearish: prefer positive K (overextended = more room to fall)
            curvature_score = float(np.clip(0.5 + k * 0.4, 0.0, 1.0))
        else:
            curvature_score = 0.5

        components.append(MFSComponent(
            name="Market_Curvature",
            score=curvature_score,
            weight=0.25,
            evidence_level=curvature_result.evidence_level,
            source="curvature",
            notes=f"K={k:.3f} | State={curvature_result.curvature_state} | {curvature_result.notes}"
        ))

    # ── LAYER 4: Multi-timeframe alignment — weight 0.15 ─────────────────────
    if multi_tf:
        # Multi-TF ν adds conviction weight
        mt_nu = multi_tf.total_nu
        mt_dir = multi_tf.total_direction
        mt_score = float(np.clip(0.5 + mt_dir * mt_nu * 0.5, 0.0, 1.0))

        components.append(MFSComponent(
            name="MultiTF_Alignment",
            score=mt_score,
            weight=0.15,
            evidence_level="TESTABLE",
            source="multi_tf_nu",
            notes=f"Total ν={mt_nu:.3f} | Dir={mt_dir:+.3f} | Dominant={multi_tf.dominant_timeframe}"
        ))
    else:
        # Adjust remaining weights if no multi-TF
        for c in components:
            c.weight = c.weight / 0.85  # renormalise to 1.0

    # ── Compute MFS ───────────────────────────────────────────────────────────
    if components:
        total_weight = sum(c.weight for c in components)
        weighted_sum = sum(c.score * c.weight for c in components)
        profile.mfs = float(np.clip(weighted_sum / total_weight, 0.0, 1.0))
    else:
        profile.mfs = 0.5

    profile.components = components

    # ── MFS Label ─────────────────────────────────────────────────────────────
    if profile.mfs >= 0.70:
        profile.mfs_label = "ZPB"
    elif profile.mfs >= 0.55:
        profile.mfs_label = "DRIFT"
    elif profile.mfs >= 0.35:
        profile.mfs_label = "IN_LOADING"
    else:
        profile.mfs_label = "DISRUPTED"

    # ── Trade Tier Recommendation ─────────────────────────────────────────────
    profile.trade_tier, profile.tier_rationale = _trade_tier(profile)

    # ── Interpretation ────────────────────────────────────────────────────────
    dir_word = "bullish" if direction > 0.1 else "bearish" if direction < -0.1 else "neutral"
    state_descriptions = {
        "ZPB":        f"Coherent field — {dir_word} bias with scale alignment. Trend-follow conditions.",
        "DRIFT":      f"Transitional field — {dir_word} lean but partial alignment. Reduce size, watch for confirmation.",
        "IN_LOADING": "Compressed field — scales diverging, energy suppressed. Volatility expansion likely. Wait or take starter only.",
        "SIC":        "Coherence collapse — forced movement conditions. SIC event in progress or imminent. Extreme caution."
    }
    profile.interpretation = state_descriptions.get(
        nu_result.field_state,
        f"Field state: {nu_result.field_state}. MFS={profile.mfs:.2f}."
    )

    if profile.late_move_break:
        profile.interpretation += f" LSII-Price [{lsii_result.flag_level}]: late-move arc breaking — structural integrity flagged."

    if curvature_result and curvature_result.curvature_state == "OVEREXTENDED":
        profile.interpretation += f" Curvature overextended (K={profile.k:.3f}) — reversal pressure loading."

    # ── Field Note ────────────────────────────────────────────────────────────
    profile.field_note = (
        f"MFS {profile.mfs:.2f} | {profile.mfs_label} | "
        f"ν={nu:.3f} [{nu_result.field_state}] | "
        f"Dir={direction:+.3f} | "
        f"Tier={profile.trade_tier}"
    )

    # ── Evidence Notes ────────────────────────────────────────────────────────
    profile.evidence_notes = [
        "ν coherence as predictive signal: [TESTABLE] — theoretical basis from BGR. Empirical validation via backtesting required.",
        "LSII-Price: [TESTABLE] — translated from KindPath Q. Requires instrument-specific calibration.",
        "Market Curvature (MA proxy): [TESTABLE] — directional, not absolute valuation.",
        "This profile does not claim to predict price. It describes measurable field conditions.",
    ]

    # ── Recommendations ───────────────────────────────────────────────────────
    if profile.trade_tier == "WAIT":
        profile.recommendations.append(
            "Field coherence insufficient for high-conviction entry. Wait for ν to rebuild above 0.40."
        )
    if profile.late_move_break:
        profile.recommendations.append(
            f"LSII-Price {lsii_result.flag_level}: move integrity flagged. Consider tightening stops or reducing size."
        )
    if curvature_result and curvature_result.curvature_state == "SYNTROPIC":
        profile.recommendations.append(
            f"Curvature syntropic (K={profile.k:.3f}) — asset trading below value estimate. "
            "Favourable for LARGE tier impact-first positioning. Confirm with KEPE WFS before entry."
        )
    if nu_result.field_state == "IN_LOADING":
        profile.recommendations.append(
            "IN-Loading detected: suppressed volatility with scale divergence. "
            "Breakout likely. Prepare for position on ν recovery above 0.40."
        )

    return profile


def _trade_tier(profile: BMRProfile) -> tuple:
    """
    Recommend trade tier based on MFS, ν, and field state.
    Returns (tier, rationale).
    """
    nu = profile.nu
    mfs = profile.mfs
    state = profile.field_state

    if state == "SIC":
        return "WAIT", "SIC event — coherence collapse, no directional conviction"

    if state == "IN_LOADING":
        if abs(profile.direction) > 0.5:
            return "NANO", "IN-Loading with directional lean — nano only pending ν recovery"
        return "WAIT", "IN-Loading with no clear direction — wait for breakout confirmation"

    if state == "DRIFT":
        if mfs >= 0.60 and nu >= 0.50:
            return "MID", "DRIFT with reasonable coherence — mid-tier appropriate"
        return "NANO", "DRIFT state — nano only, wait for ZPB confirmation"

    if state == "ZPB":
        if nu >= 0.80 and mfs >= 0.70:
            return "LARGE", "ZPB + high ν + strong MFS — large tier impact-first eligible"
        if nu >= 0.65:
            return "MID", "ZPB state, moderate ν — mid-tier appropriate"
        return "NANO", "ZPB state but ν below threshold — nano with scale-in plan"

    return "NANO", f"Default: {state} state — conservative approach"
