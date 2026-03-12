# AI Agent Rules for kindpath-bmr

## Session Init Protocol

Before reading code or making changes, run:
```bash
cat ~/.kindpath/HANDOVER.md
python3 ~/.kindpath/kp_memory.py dump --domain gotcha
python3 ~/.kindpath/kp_memory.py dump
```

---

## What This Is

KindPath Behavioural Market Relativity (BMR) engine. Measures benevolence as
a core economic metric and manages sovereign identity validation.

## Structure

```
bmr_server.py       — Flask API server
core/               — BMR computation engine
feeds/feeds.py      — Data feed integrations
ndis/               — NDIS-specific BMR tools
db/                 — SQLite registry (encrypted)
```

## Operational Commands

- **Install**: `pip install -r requirements.txt`
- **Run**: `python bmr_server.py`
- **Test**: `pytest`
- **Docker**: `docker build -t kindpath-bmr . && docker run --env-file .env kindpath-bmr`

## Rules

- BMR scores are calculated, not assigned — no manual overrides
- Registry entry data must be encrypted at rest
- Validation logic in `core/` is the source of truth for benevolence claims
- Read `doctrine.md` before modifying scoring logic

## Security Mandates

- Protect secrets: no keys or biometric templates in source
- All registry data encrypted
- No PII in logs

## Note

This repo is archived on GitHub. Changes remain local unless the repo is unarchived.
