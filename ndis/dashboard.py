import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime

# Paths
ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "db" / "ndis_clients.db"
if not DB_PATH.parent.exists() or not DB_PATH.exists():
    DB_PATH = Path("db/ndis_clients.db")

st.set_page_config(page_title="KindAI NDIS Dashboard", layout="wide")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_data(table_name):
    with get_connection() as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

st.title("🛡️ KindAI Care Management")
st.subheader("ShiftCare-lite Environment")

# Sidebar
st.sidebar.image("https://kindpathcollective.org/wp-content/uploads/2023/05/KP-Logo-Dark-300x300.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Roster", "Clients", "Staff", "Compliance Audit"])

if page == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    
    with get_connection() as conn:
        c_count = conn.execute("SELECT COUNT(*) FROM clients WHERE active=1").fetchone()[0]
        w_count = conn.execute("SELECT COUNT(*) FROM workers WHERE active=1").fetchone()[0]
        s_count = conn.execute("SELECT COUNT(*) FROM shifts WHERE start_time > datetime('now') AND status='SCHEDULED'").fetchone()[0]
        i_count = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]

    col1.metric("Active Clients", c_count)
    col2.metric("Support Staff", w_count)
    col3.metric("Upcoming Shifts", s_count)
    col4.metric("Total Incidents", i_count)

    st.divider()
    
    # Recent Shifts
    st.subheader("Recent & Upcoming Shifts")
    with get_connection() as conn:
        shifts_df = pd.read_sql_query("""
            SELECT s.id, c.name as Client, w.name as Worker, s.start_time, s.end_time, s.status
            FROM shifts s
            JOIN clients c ON s.client_id = c.id
            JOIN workers w ON s.worker_id = w.id
            ORDER BY s.start_time DESC LIMIT 10
        """, conn)
    st.table(shifts_df)

elif page == "Roster":
    st.subheader("Weekly Roster")
    with get_connection() as conn:
        roster_df = pd.read_sql_query("""
            SELECT s.start_time, s.end_time, c.name as Client, w.name as Worker, s.status
            FROM shifts s
            JOIN clients c ON s.client_id = c.id
            JOIN workers w ON s.worker_id = w.id
            WHERE s.start_time >= date('now', '-7 days')
            ORDER BY s.start_time ASC
        """, conn)
    
    if not roster_df.empty:
        roster_df['start_time'] = pd.to_datetime(roster_df['start_time'])
        fig = px.timeline(roster_df, x_start="start_time", x_end="end_time", y="Worker", color="Client", text="Client")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(roster_df, use_container_width=True)
    else:
        st.info("No shifts found for the current period.")

elif page == "Clients":
    st.subheader("Participant Directory")
    clients = load_data("clients")
    st.dataframe(clients, use_container_width=True)
    
    # Budget Visualization
    if not clients.empty:
        st.subheader("Budget Utilization")
        clients['total_budget'] = clients['budget_core'] + clients['budget_cb']
        clients['total_spent'] = clients['spent_core'] + clients['spent_cb']
        fig = px.bar(clients, x="name", y=["total_spent", "total_budget"], barmode="group", title="Spent vs Budget")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Staff":
    st.subheader("Support Workers")
    workers = load_data("workers")
    st.dataframe(workers, use_container_width=True)

elif page == "Compliance Audit":
    st.subheader("NDIS Legislative Audit")
    
    from ndis.tools import NDISTools
    tools = NDISTools()
    
    if st.button("Run Full System Audit"):
        with st.spinner("Analyzing service delivery against NDIS Legislation..."):
            result = tools.handle("audit")
            st.markdown(result)
            
    st.divider()
    st.subheader("Missing Progress Notes")
    with get_connection() as conn:
        missing = pd.read_sql_query("""
            SELECT s.id as Shift_ID, c.name as Client, w.name as Worker, s.start_time
            FROM shifts s
            JOIN clients c ON s.client_id = c.id
            JOIN workers w ON s.worker_id = w.id
            WHERE s.status = 'COMPLETED' 
            AND s.id NOT IN (SELECT shift_id FROM progress_notes WHERE shift_id IS NOT NULL)
        """, conn)
    if not missing.empty:
        st.warning(f"Found {len(missing)} completed shifts without progress notes.")
        st.dataframe(missing, use_container_width=True)
    else:
        st.success("All completed shifts have associated progress notes.")
