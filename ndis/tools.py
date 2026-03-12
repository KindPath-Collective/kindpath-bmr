"""
NDIS Business Tooling — Care Management System
==============================================
A 'ShiftCare-lite' environment for NDIS support coordination and delivery.

Commands via Telegram /ndis or CLI:
  status              — Dashboard
  worker add/list     — Manage staff
  client add/list     — Manage participants
  shift add/start/end — Roster & Time/Attendance
  roster [days]       — View schedule
  incident log        — Report incidents
  price <query>       — Search Price Guide
  audit [name]        — Compliance & Legislative check
  consult <query>     — Advisory Council
  proposal/gap/invoice — Business tools

Data: /Users/sam/kindai/db/ndis_clients.db
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from dateutil import parser as date_parser

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "db" / "ndis_clients.db"
if not DB_PATH.parent.exists() or not DB_PATH.exists():
    DB_PATH = Path("db/ndis_clients.db")
PRICE_GUIDE_PATH = Path(__file__).parent / "price_guide.json"

logger = logging.getLogger(__name__)

NDIS_SYSTEM = """You are a specialist NDIS support coordinator and business advisor.
You understand NDIS plans, support categories (Core, Capacity Building, Capital),
cancellation policies, and compliance. Be practical and specific. No filler."""


class NDISTools:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.pricing = self._load_pricing()

    def _load_pricing(self) -> List[Dict[str, Any]]:
        if PRICE_GUIDE_PATH.exists():
            try:
                return json.loads(PRICE_GUIDE_PATH.read_text())
            except Exception as e:
                logger.error(f"Failed to load price guide: {e}")
        return []

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            # 1. Clients
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clients (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    ndis_number TEXT,
                    plan_start  TEXT,
                    plan_end    TEXT,
                    budget_core REAL DEFAULT 0,
                    budget_cb   REAL DEFAULT 0,
                    budget_cap  REAL DEFAULT 0,
                    spent_core  REAL DEFAULT 0,
                    spent_cb    REAL DEFAULT 0,
                    spent_cap   REAL DEFAULT 0,
                    goals       TEXT,
                    notes       TEXT,
                    active      INTEGER DEFAULT 1,
                    updated     TEXT DEFAULT (datetime('now'))
                );
            """)
            
            # 2. Support Workers
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workers (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    role        TEXT DEFAULT 'Support Worker',
                    phone       TEXT,
                    email       TEXT,
                    base_rate   REAL DEFAULT 0,
                    active      INTEGER DEFAULT 1,
                    notes       TEXT
                );
            """)

            # 3. Shifts (Roster & Attendance)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shifts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id   INTEGER REFERENCES clients(id),
                    worker_id   INTEGER REFERENCES workers(id),
                    start_time  TEXT NOT NULL, -- Scheduled Start
                    end_time    TEXT NOT NULL, -- Scheduled End
                    actual_start TEXT,         -- Clock In
                    actual_end   TEXT,         -- Clock Out
                    status      TEXT DEFAULT 'SCHEDULED', -- SCHEDULED, IN_PROGRESS, COMPLETED, CANCELLED
                    support_item TEXT,
                    notes       TEXT,
                    invoiced    INTEGER DEFAULT 0
                );
            """)

            # 4. Incidents
            conn.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id   INTEGER REFERENCES clients(id),
                    worker_id   INTEGER REFERENCES workers(id),
                    date        TEXT DEFAULT (datetime('now')),
                    severity    TEXT, -- LOW, MEDIUM, CRITICAL
                    description TEXT,
                    actions     TEXT,
                    reported_at TEXT DEFAULT (datetime('now'))
                );
            """)

            # 5. Progress Notes (linked to shifts)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS progress_notes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    shift_id    INTEGER REFERENCES shifts(id),
                    client_id   INTEGER REFERENCES clients(id),
                    worker_id   INTEGER REFERENCES workers(id),
                    content     TEXT,
                    created_at  TEXT DEFAULT (datetime('now'))
                );
            """)
            
            # Legacy sessions table (kept for backward compat)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id   INTEGER REFERENCES clients(id),
                    date        TEXT NOT NULL,
                    hours       REAL,
                    support_cat TEXT,
                    rate        REAL,
                    notes       TEXT,
                    invoiced    INTEGER DEFAULT 0
                );
            """)

    def _ai(self, prompt: str) -> str:
        try:
            from kindai_client import KindAIClient
            ai = KindAIClient(system=NDIS_SYSTEM)
            return ai.ask(prompt)
        except Exception as e:
            return f"[AI unavailable: {e}]"

    def handle(self, command: str) -> str:
        parts = command.strip().split()
        if not parts:
            return self._status()

        cmd = parts[0].lower()
        args = parts[1:]

        # Dispatcher
        if cmd == "status":
            return self._status()
        elif cmd == "worker":
            return self._handle_worker(args)
        elif cmd == "client":
            return self._handle_client(args)
        elif cmd == "shift":
            return self._handle_shift(args)
        elif cmd == "roster":
            days = int(args[0]) if args and args[0].isdigit() else 7
            return self._roster_view(days)
        elif cmd == "incident":
            return self._handle_incident(args)
        elif cmd == "price":
            return self._price_lookup(" ".join(args))
        elif cmd == "audit":
            return self._legislative_audit(args[0] if args else "")
        elif cmd == "consult":
            return self._consult_council(" ".join(args))
        elif cmd == "proposal":
            return self._proposal(args[0] if args else "", " ".join(args[1:]) if len(args) > 1 else "")
        elif cmd == "gap":
            return self._gap(args[0] if args else "")
        elif cmd == "invoice":
            return self._invoice(args[0] if args else "")
        elif cmd == "cancel":
            return self._cancel_check(args[0] if args else "", " ".join(args[1:]) if len(args) > 1 else "")
        elif cmd == "add": # Legacy alias
            return "Use `/ndis client add` or `/ndis worker add` instead."
        else:
            return self._help()

    def _help(self) -> str:
        return (
            "*NDIS Care Management Commands*\n"
            "• `status` — Dashboard\n"
            "• `worker add <name> <role> <rate>`\n"
            "• `worker list`\n"
            "• `client add <name> <ndis_num>`\n"
            "• `client list`\n"
            "• `shift add <client_id> <worker_id> <start> <end>`\n"
            "• `shift start <shift_id>`\n"
            "• `shift end <shift_id> [notes]`\n"
            "• `roster [days]`\n"
            "• `incident log <client_id> <details>`\n"
            "• `price <query>`\n"
            "• `consult <question>`\n"
        )

    # ─── Sub-Handlers ─────────────────────────────────────────────

    def _handle_worker(self, args: List[str]) -> str:
        if not args: return "Usage: worker list | worker add <name> <role> <rate>"
        sub = args[0].lower()
        
        if sub == "list":
            return self._list_workers()
        elif sub == "add" and len(args) >= 4:
            # Reconstruct name from parts (heuristic: assume rate is last, role is 2nd last?)
            # Better: worker add Name Role Rate. Let's keep it simple for CLI.
            # worker add Alice SW 35.50
            rate = float(args[-1])
            role = args[-2]
            name = " ".join(args[1:-2])
            return self._add_worker(name, role, rate)
        return "Usage: worker add <name> <role> <rate>"

    def _handle_client(self, args: List[str]) -> str:
        if not args: return "Usage: client list | client add <name> <ndis_num>"
        sub = args[0].lower()
        
        if sub == "list":
            return self._list_clients()
        elif sub == "add" and len(args) >= 3:
            ndis_num = args[-1]
            name = " ".join(args[1:-1])
            return self._add_client(name, ndis_num)
        return "Usage: client add <name> <ndis_num>"

    def _handle_shift(self, args: List[str]) -> str:
        if not args: return "Usage: shift add | shift start | shift end"
        sub = args[0].lower()
        
        if sub == "add" and len(args) >= 5:
            # shift add cid wid 2026-03-09T09:00 2026-03-09T12:00
            return self._add_shift(args[1], args[2], args[3], args[4])
        elif sub == "start" and len(args) >= 2:
            return self._clock_shift(args[1], "start")
        elif sub == "end" and len(args) >= 2:
            notes = " ".join(args[2:]) if len(args) > 2 else ""
            return self._clock_shift(args[1], "end", notes)
        return "Usage: shift add <cid> <wid> <start> <end> | shift start <id> | shift end <id> <notes>"

    def _handle_incident(self, args: List[str]) -> str:
        if len(args) < 3 or args[0] != "log":
            return "Usage: incident log <client_id> <details>"
        
        client_id = args[1]
        details = " ".join(args[2:])
        return self._log_incident(client_id, details)

    # ─── Core Logic ───────────────────────────────────────────────

    def _status(self) -> str:
        with self._conn() as conn:
            c_count = conn.execute("SELECT COUNT(*) FROM clients WHERE active=1").fetchone()[0]
            w_count = conn.execute("SELECT COUNT(*) FROM workers WHERE active=1").fetchone()[0]
            s_count = conn.execute("SELECT COUNT(*) FROM shifts WHERE start_time > datetime('now') AND status='SCHEDULED'").fetchone()[0]
            i_count = conn.execute("SELECT COUNT(*) FROM incidents WHERE date > date('now', '-7 days')").fetchone()[0]
        
        return (
            f"*ShiftCare-lite Dashboard*\n"
            f"👥 Clients: {c_count}\n"
            f"👷 Staff: {w_count}\n"
            f"📅 Upcoming Shifts: {s_count}\n"
            f"⚠️ Recent Incidents: {i_count}\n"
            f"💡 _Tip: Use /ndis roster to view schedule_"
        )

    def _add_worker(self, name: str, role: str, rate: float) -> str:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO workers (name, role, base_rate) VALUES (?, ?, ?)",
                (name, role, rate)
            )
        return f"✅ Added worker: {name} ({role}) @ ${rate}/hr (ID: {cur.lastrowid})"

    def _list_workers(self) -> str:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM workers WHERE active=1").fetchall()
        if not rows: return "No workers found."
        lines = ["*Support Workers*"]
        for r in rows:
            lines.append(f"• [ID:{r['id']}] {r['name']} ({r['role']})")
        return "\n".join(lines)

    def _add_client(self, name: str, ndis_num: str) -> str:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO clients (name, ndis_number) VALUES (?, ?)",
                (name, ndis_num)
            )
        return f"✅ Added client: {name} (ID: {cur.lastrowid})"

    def _list_clients(self) -> str:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM clients WHERE active=1").fetchall()
        if not rows: return "No clients found."
        lines = ["*Participants*"]
        for r in rows:
            lines.append(f"• [ID:{r['id']}] {r['name']}")
        return "\n".join(lines)

    def _add_shift(self, cid: str, wid: str, start: str, end: str) -> str:
        try:
            # Robust parsing
            s_dt = date_parser.parse(start)
            e_dt = date_parser.parse(end)
            if s_dt.tzinfo is None: s_dt = s_dt.replace(tzinfo=None) # naive
            if e_dt.tzinfo is None: e_dt = e_dt.replace(tzinfo=None)
            
            with self._conn() as conn:
                cur = conn.execute(
                    "INSERT INTO shifts (client_id, worker_id, start_time, end_time) VALUES (?, ?, ?, ?)",
                    (cid, wid, s_dt.isoformat(), e_dt.isoformat())
                )
            return f"✅ Shift scheduled (ID: {cur.lastrowid}): {s_dt.strftime('%a %H:%M')} - {e_dt.strftime('%H:%M')}"
        except Exception as e:
            return f"❌ Error adding shift: {e}"

    def _clock_shift(self, sid: str, action: str, notes: str = "") -> str:
        with self._conn() as conn:
            shift = conn.execute("SELECT * FROM shifts WHERE id=?", (sid,)).fetchone()
            if not shift: return "Shift not found."
            
            now = datetime.now().isoformat()
            if action == "start":
                conn.execute("UPDATE shifts SET status='IN_PROGRESS', actual_start=? WHERE id=?", (now, sid))
                return f"🕒 Shift {sid} STARTED at {now[11:16]}"
            elif action == "end":
                conn.execute("UPDATE shifts SET status='COMPLETED', actual_end=?, notes=? WHERE id=?", (now, notes, sid))
                # Also log note if provided
                if notes:
                    conn.execute(
                        "INSERT INTO progress_notes (shift_id, client_id, worker_id, content) VALUES (?,?,?,?)",
                        (sid, shift["client_id"], shift["worker_id"], notes)
                    )
                return f"✅ Shift {sid} COMPLETED at {now[11:16]}"
        return "Invalid action."

    def _roster_view(self, days: int = 7) -> str:
        with self._conn() as conn:
            shifts = conn.execute("""
                SELECT s.*, c.name as cname, w.name as wname 
                FROM shifts s
                JOIN clients c ON s.client_id = c.id
                JOIN workers w ON s.worker_id = w.id
                WHERE s.start_time >= datetime('now', '-1 day') 
                ORDER BY s.start_time ASC LIMIT 20
            """).fetchall()
            
        if not shifts: return "No upcoming shifts."
        lines = [f"*Roster (Upcoming)*"]
        last_date = ""
        for s in shifts:
            dt = date_parser.parse(s['start_time'])
            date_str = dt.strftime("%A %d %b")
            if date_str != last_date:
                lines.append(f"\n*{date_str}*")
                last_date = date_str
            
            status_icon = "🟢" if s['status'] == 'COMPLETED' else "🔵" if s['status'] == 'IN_PROGRESS' else "⚪️"
            lines.append(f"{status_icon} {dt.strftime('%H:%M')} [ID:{s['id']}] {s['cname']} ({s['wname']})")
            
        return "\n".join(lines)

    def _log_incident(self, cid: str, details: str) -> str:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO incidents (client_id, description, severity) VALUES (?, ?, 'MEDIUM')",
                (cid, details)
            )
        return f"⚠️ Incident logged for Client ID {cid}."

    def _price_lookup(self, query: str) -> str:
        if not self.pricing:
            return "Price guide not loaded."
        
        matches = []
        q = query.lower()
        for item in self.pricing:
            if q in item['item'].lower() or q in item['code'].lower():
                matches.append(item)
        
        if not matches: return "No matching items found."
        
        lines = [f"*Price Guide Matches for '{query}'*"]
        for m in matches[:5]:
            lines.append(f"• `{m['code']}`: ${m['price']}/{m['unit']} — {m['item']}")
        return "\n".join(lines)


    def _legislative_audit(self, name: str = "") -> str:
        """
        Audit service delivery against NDIS Practice Standards & Legislation.
        Checks for: missing notes, budget velocity, and compliance gaps.
        """
        with self._conn() as conn:
            if name:
                clients = conn.execute("SELECT * FROM clients WHERE name LIKE ? AND active=1", (f"%{name}%",)).fetchall()
            else:
                clients = conn.execute("SELECT * FROM clients WHERE active=1").fetchall()

            if not clients: return "No active clients found to audit."

            audit_results = []
            for c in clients:
                # 1. Check for shifts without progress notes
                missing_notes = conn.execute("""
                    SELECT COUNT(*) FROM shifts 
                    WHERE client_id = ? AND status = 'COMPLETED' 
                    AND id NOT IN (SELECT shift_id FROM progress_notes WHERE shift_id IS NOT NULL)
                """, (c['id'],)).fetchone()[0]

                # 2. Budget Velocity
                budget_total = (c['budget_core'] or 0) + (c['budget_cb'] or 0)
                spent_total = (c['spent_core'] or 0) + (c['spent_cb'] or 0)
                usage_pct = (spent_total / budget_total * 100) if budget_total > 0 else 0

                # 3. Incident count
                incidents = conn.execute("SELECT COUNT(*) FROM incidents WHERE client_id = ?", (c['id'],)).fetchone()[0]

                audit_results.append({
                    "name": c['name'],
                    "missing_notes": missing_notes,
                    "usage_pct": usage_pct,
                    "incidents": incidents,
                    "plan_end": c['plan_end']
                })

        # AI Legislative Assessment
        prompt = (
            "You are an NDIS Quality and Safeguards auditor. Review the following service delivery data:

"
            + "
".join([f"- Client {r['name']}: {r['missing_notes']} completed shifts missing progress notes. "
                         f"Budget used: {r['usage_pct']:.1f}%. Incidents: {r['incidents']}. Plan ends: {r['plan_end']}."
                         for r in audit_results]) +
            "

Identify specific risks against the NDIS Practice Standards (e.g., Provision of Supports, "
            "Governance and Operational Management) and relevant sections of the NDIS Act 2013. "
            "Provide 3-4 high-priority corrective actions."
        )
        
        assessment = self._ai(prompt)
        return f"*NDIS Legislative Audit*

{assessment}"

    def _consult_council(self, question: str) -> str:
        try:
            # Lazy import to avoid circular dependency
            import sys
            sys.path.insert(0, str(ROOT))
            from advisory.council import AdvisoryCouncil
            
            council = AdvisoryCouncil()
            return council.deliberate(question)
        except Exception as e:
            return f"Council unavailable: {e}"

    # ─── Legacy/Business Logic ────────────────────────────────────

    def _proposal(self, name: str, context: str) -> str:
        # (Simplified for brevity, assumes client exists or basic prompt)
        return self._ai(f"Draft NDIS proposal for {name}. Context: {context}")

    def _gap(self, name: str) -> str:
        return self._ai(f"Gap analysis for {name}")

    def _invoice(self, name: str) -> str:
        with self._conn() as conn:
            # Find client
            client = conn.execute("SELECT * FROM clients WHERE name LIKE ? LIMIT 1", (f"%{name}%",)).fetchone()
            if not client: return "Client not found."
            
            # Find completed uninvoiced shifts
            shifts = conn.execute(
                "SELECT * FROM shifts WHERE client_id=? AND status='COMPLETED' AND invoiced=0",
                (client['id'],)
            ).fetchall()
            
            if not shifts: return f"No billable shifts for {client['name']}."
            
            total = 0.0
            lines = [f"*Invoice Draft — {client['name']}*"]
            for s in shifts:
                start = date_parser.parse(s['actual_start'] or s['start_time'])
                end = date_parser.parse(s['actual_end'] or s['end_time'])
                duration = (end - start).total_seconds() / 3600
                # Get worker rate (simplified)
                w = conn.execute("SELECT base_rate FROM workers WHERE id=?", (s['worker_id'],)).fetchone()
                rate = w['base_rate'] if w else 50.0
                cost = duration * rate
                total += cost
                lines.append(f"• {start.strftime('%d/%m')} {duration:.1f}h @ ${rate} = ${cost:.2f}")
            
            lines.append(f"\n*Total: ${total:.2f}*")
            return "\n".join(lines)

    def _cancel_check(self, name: str, reason: str) -> str:
        return self._ai(f"Check cancellation policy for {name}. Reason: {reason}")
