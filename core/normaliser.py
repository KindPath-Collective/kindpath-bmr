"""
BMR — Scale Normaliser
=======================
Aggregates multiple raw signals within each scale layer
into a single weighted directional value: -1.0 → +1.0

Each scale has its own weighting table.
Confidence-weighted average within each source group.
Evidence level modulates confidence ceiling.
"""

from __future__ import annotations
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict
from feeds.feeds import RawSignal

logger = logging.getLogger(__name__)

# Evidence level confidence ceilings
EVIDENCE_CEILING = {
    "ESTABLISHED":  1.00,
    "TESTABLE":     0.75,
    "SPECULATIVE":  0.40,
}

# Source weights within each scale
PARTICIPANT_WEIGHTS = {
    "momentum":        0.35,
    "volume_pressure": 0.30,
    "options_skew":    0.35,
}

INSTITUTIONAL_WEIGHTS = {
    "cot":           0.40,
    "inst_flow":     0.30,
    "credit_spread": 0.30,
}

SOVEREIGN_WEIGHTS = {
    "macro_fred":   0.40,
    "central_bank": 0.35,
    "geopolitical": 0.25,
}

SCALE_WEIGHTS = {
    "PARTICIPANT":  PARTICIPANT_WEIGHTS,
    "INSTITUTIONAL": INSTITUTIONAL_WEIGHTS,
    "SOVEREIGN":    SOVEREIGN_WEIGHTS,
}


@dataclass
class ScaleReading:
    """Aggregated directional reading for one scale."""
    scale: str
    value: float           # -1.0 → +1.0
    confidence: float      # 0.0 → 1.0
    source_count: int
    source_detail: Dict[str, float] = field(default_factory=dict)


def normalise_scale(signals: List[RawSignal], scale: str) -> ScaleReading:
    """
    Aggregate raw signals for a given scale into a single reading.
    Applies evidence ceiling, source weighting, confidence weighting.
    """
    if not signals:
        return ScaleReading(
            scale=scale, value=0.0, confidence=0.0, source_count=0
        )

    weights = SCALE_WEIGHTS.get(scale, {})
    weighted_sum = 0.0
    weight_total = 0.0
    source_detail = {}

    for sig in signals:
        if sig.scale != scale:
            continue

        # Apply evidence ceiling to confidence
        ceiling = EVIDENCE_CEILING.get(sig.evidence_level, 0.5)
        effective_conf = sig.confidence * ceiling

        # Source weight
        source_w = weights.get(sig.source, 0.20)  # default 0.20 for unknown sources

        # Combined weight
        w = source_w * effective_conf
        weighted_sum += sig.value * w
        weight_total += w
        source_detail[sig.source] = sig.value

    if weight_total < 1e-10:
        return ScaleReading(
            scale=scale, value=0.0, confidence=0.0,
            source_count=len(signals), source_detail=source_detail
        )

    value = float(np.clip(weighted_sum / weight_total, -1, 1))
    confidence = float(np.clip(weight_total / sum(weights.values()), 0, 1))

    return ScaleReading(
        scale=scale,
        value=value,
        confidence=confidence,
        source_count=len(signals),
        source_detail=source_detail,
    )
