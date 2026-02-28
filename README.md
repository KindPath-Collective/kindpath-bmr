# Behavioural Market Relativity (BMR)
## KindPath Trading Engine — Core Signal Layer

```
M = [(Participant × Institutional × Sovereign) · ν]²
```

BMR is the market-side signal layer for the DFTE (Dual Field Trading Engine).
It reads the price field the same way KindEarth reads the world field —
as a coherence system, not a prediction machine.

---

## Architecture

```
feeds/          Raw data ingestors (market data, COT, options, macro)
core/
  normaliser.py     Scale signal normalisation → [-1, +1]
  nu_engine.py      ν coherence computation across scales + timeframes
  lsii_price.py     Late-Move Inversion Index (translated from KindPath Q)
  field_state.py    ZPB/DRIFT/IN-Loading/SIC classifier
  curvature.py      Market Curvature Index (tokenisation gap)
  bmr_profile.py    Full BMR field reading synthesiser
  mfs.py            Market Field Score output
tests/
bmr_server.py       FastAPI signal server (same pattern as q_server.py)
```

---

## Field States

| State | ν | Meaning |
|-------|---|---------|
| ZPB — Coherent Trend | > 0.75 | All scales aligned, M amplifying |
| DRIFT — Transition | 0.40–0.75 | Partial alignment, weakening |
| IN-Loading — Compression | 0.15–0.40 | Scales diverging, pressure building |
| SIC — Event | < 0.15 | Coherence collapse, forced movement |

---

## Integration

BMR (Market Field Score) + KEPE (World Field Score) → DFTE trade selection
