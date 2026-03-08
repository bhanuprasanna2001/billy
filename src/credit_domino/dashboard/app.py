"""Credit Domino Dashboard — 3-tab Streamlit app with scoring, simulation viz, analytics."""

import os

import httpx
import pandas as pd
import streamlit as st
from pyvis.network import Network

API_BASE = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Domino", layout="wide", page_icon="🎲")
st.title("Credit Domino — Risk Scoring & Contagion Simulator")


# ── Tab layout ──────────────────────────────────────────────────────────────

tab_score, tab_domino, tab_analytics = st.tabs(
    ["🎯 Score Customer", "🃏 Domino Simulation", "📊 Analytics"]
)


# ─── Tab 1: Score Customer ──────────────────────────────────────────────────

with tab_score:
    st.header("Real-time Credit Scoring")

    with st.form("score_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            customer_id = st.text_input("Customer ID", "PROSPER_42")
            person_age = st.number_input("Age", 18, 100, 30)
            person_income = st.number_input("Income", 0, 500000, 50000)
            person_home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "OWN", "MORTGAGE", "OTHER"],
            )
        with col2:
            person_emp_length = st.number_input("Employment Length (yrs)", 0.0, 50.0, 5.0)
            loan_intent = st.selectbox(
                "Loan Intent",
                [
                    "PERSONAL",
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "HOMEIMPROVEMENT",
                    "DEBTCONSOLIDATION",
                ],
            )
            loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
            loan_amnt = st.number_input("Loan Amount", 0, 100000, 10000)
        with col3:
            loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 11.0)
            loan_percent_income = st.number_input("Loan % Income", 0.0, 1.0, 0.2)
            cb_default = st.selectbox("Prior Default on File?", [0, 1])
            cb_cred_hist = st.number_input("Credit History Length", 0, 50, 4)

        submitted = st.form_submit_button("Score", type="primary")

    if submitted:
        payload = {
            "customer_id": customer_id,
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_default,
            "cb_person_cred_hist_length": cb_cred_hist,
        }
        try:
            resp = httpx.post(f"{API_BASE}/score", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                band = data["decision_band"]
                color = {"low": "green", "medium": "orange", "high": "red"}[band]

                c1, c2, c3 = st.columns(3)
                c1.metric("Risk Score", f"{data['risk_score']:.2%}")
                c2.metric("Decision Band", band.upper())
                c3.metric("Event ID", data["scoring_event_id"][:8] + "...")

                st.markdown(f"**Decision:** :{color}[{band.upper()} RISK]")

                st.subheader("Top Risk Factors")
                factors_df = pd.DataFrame(data["top_factors"])
                st.bar_chart(factors_df.set_index("feature")["shap_value"])
            else:
                st.error(f"API error: {resp.status_code} \u2014 {resp.text}")
        except httpx.ConnectError:
            st.error("Cannot connect to API. Is the server running on port 8000?")


# ─── Tab 2: Domino Simulation ──────────────────────────────────────────────

# Fire-gradient palette: trigger → fallen hops → stressed-only
_HOP_COLORS = {
    0: "#FF0000",  # trigger — bright red
    1: "#FF4500",  # hop 1 fallen — orange-red
    2: "#FF8C00",  # hop 2 fallen — dark orange
    3: "#FFD700",  # hop 3 fallen — gold
}
_FALLEN_DEFAULT = "#FFA500"  # hop 4+ fallen
_STRESSED_COLOR = "#4A90D9"  # reached but not fallen — steel blue

with tab_domino:
    st.header("Contagion Cascade Simulator")

    col_ctrl, col_viz = st.columns([1, 3])
    with col_ctrl:
        trigger_id = st.text_input("Trigger Customer", "PROSPER_0", key="trigger")
        initial_shock = st.slider("Initial Shock", 0.0, 1.0, 1.0)
        decay = st.slider("Decay per Hop", 0.0, 1.0, 0.6)
        threshold = st.slider("Fall Threshold", 0.0, 1.0, 0.3)
        max_hops = st.slider("Max Hops", 1, 10, 5)
        n_customers = st.number_input("Graph Size", 50, 5000, 500)
        run_sim = st.button("Run Simulation", type="primary")

    with col_viz:
        # Persist cascade data across slider changes
        if run_sim:
            payload = {
                "trigger_customer_id": trigger_id,
                "initial_shock": initial_shock,
                "decay": decay,
                "threshold": threshold,
                "max_hops": max_hops,
                "n_customers": n_customers,
            }
            try:
                with st.spinner("Running cascade simulation..."):
                    resp = httpx.post(
                        f"{API_BASE}/simulate-domino",
                        json=payload,
                        timeout=60,
                    )
                if resp.status_code == 200:
                    st.session_state["cascade_data"] = resp.json()
                elif resp.status_code == 404:
                    st.warning(resp.json()["detail"])
                else:
                    st.error(f"API error: {resp.status_code}")
            except httpx.ConnectError:
                st.error("Cannot connect to API. Start the server first.")

        data = st.session_state.get("cascade_data")
        if data:
            cascade = data["cascade"]
            max_hop_reached = data["max_hop"]

            # ── Summary metrics ──
            m1, m2, m3 = st.columns(3)
            m1.metric("Affected", data["total_affected"])
            m2.metric("Fallen", data["total_fallen"])
            m3.metric("Max Hop", max_hop_reached)

            # ── Hop slider for step-by-step cascade viewing ──
            show_hops = st.slider(
                "Cascade Depth",
                min_value=0,
                max_value=max(max_hop_reached, 1),
                value=max_hop_reached,
                help="Step through the cascade hop by hop",
            )

            # ── Color legend ──
            legend_items = [
                ("#FF0000", "Trigger"),
                ("#FF4500", "Fallen (hop 1)"),
                ("#FF8C00", "Fallen (hop 2)"),
                ("#FFD700", "Fallen (hop 3+)"),
                ("#4A90D9", "Stressed (not fallen)"),
            ]
            legend_html = " &nbsp; ".join(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'background:{color};border-radius:50%;vertical-align:middle;">'
                f'</span> <span style="font-size:0.85em;">{label}</span>'
                for color, label in legend_items
            )
            st.markdown(legend_html, unsafe_allow_html=True)

            # ── Filter cascade to current slider depth ──
            visible = [c for c in cascade if c["hop"] <= show_hops]

            # ── Build pyvis network ──
            net = Network(
                height="550px",
                width="100%",
                directed=True,
                bgcolor="#0e1117",
                font_color="#ffffff",
                notebook=False,
                select_menu=False,
                filter_menu=False,
            )

            trigger_cid = data["trigger_customer_id"]
            visible_ids = {c["customer_id"] for c in visible}

            for c in visible:
                cid = c["customer_id"]
                hop = c["hop"]
                fallen = c["fallen"]

                if cid == trigger_cid:
                    color = _HOP_COLORS[0]
                    size = 40
                elif fallen:
                    color = _HOP_COLORS.get(hop, _FALLEN_DEFAULT)
                    size = 15 + c["stress"] * 20
                else:
                    color = _STRESSED_COLOR
                    size = 12

                net.add_node(
                    cid,
                    label=f"{cid}\n{c['stress']:.0%}",
                    color=color,
                    size=size,
                    title=(
                        f"{cid} | Hop: {hop} | Stress: {c['stress']:.4f}"
                        f" | {'FALLEN' if fallen else 'stable'}"
                    ),
                    font={"size": 10, "color": "#cccccc"},
                )

            # Add directed edges from parent → child (following cascade path)
            edge_colors = {
                "co-borrower": "#e06666",
                "guarantor": "#f6b26b",
                "employer": "#93c47d",
            }
            for c in visible:
                parent = c.get("parent")
                if parent and parent in visible_ids:
                    etype = c.get("edge_type", "unknown") or "unknown"
                    net.add_edge(
                        parent,
                        c["customer_id"],
                        title=etype,
                        color=edge_colors.get(etype, "#888888"),
                        width=1.5,
                    )

            net.set_options("""{
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -8000,
                        "centralGravity": 0.35,
                        "springLength": 150,
                        "springConstant": 0.04,
                        "damping": 0.09
                    },
                    "minVelocity": 0.75
                },
                "edges": {
                    "smooth": {"type": "curvedCW", "roundness": 0.1},
                    "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
                },
                "nodes": {
                    "borderWidth": 2,
                    "shadow": true
                }
            }""")

            html = net.generate_html()
            st.components.v1.html(html, height=580, scrolling=False)

            # ── Cascade details table ──
            with st.expander("Cascade Details"):
                st.dataframe(
                    pd.DataFrame(visible),
                    width="stretch",
                    hide_index=True,
                )


# ─── Tab 3: Analytics (ClickHouse) ──────────────────────────────────────────

with tab_analytics:
    st.header("Scoring Analytics")

    try:
        import clickhouse_connect

        from credit_domino.config import Settings

        settings = Settings()
        ch = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            database=settings.clickhouse_db,
        )

        count = ch.command("SELECT count() FROM scoring_events")
        st.metric("Total Scoring Events", count)

        if count > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Risk Band Distribution")
                band_df = ch.query_df(
                    "SELECT decision_band, count() as cnt "
                    "FROM scoring_events GROUP BY decision_band ORDER BY cnt DESC"
                )
                if not band_df.empty:
                    st.bar_chart(band_df.set_index("decision_band"))

            with col2:
                st.subheader("Hourly Scoring Volume")
                hourly_df = ch.query_df(
                    "SELECT hour, sum(event_count) AS scores, "
                    "sum(risk_sum) / sum(event_count) AS avg_risk "
                    "FROM scoring_hourly "
                    "GROUP BY hour ORDER BY hour"
                )
                if not hourly_df.empty:
                    st.line_chart(hourly_df.set_index("hour")["scores"])
                else:
                    # Fallback for pre-MV data
                    scores_df = ch.query_df(
                        "SELECT risk_score FROM scoring_events ORDER BY scored_at DESC LIMIT 1000"
                    )
                    if not scores_df.empty:
                        st.line_chart(scores_df)

            st.subheader("Recent Scoring Events")
            recent_df = ch.query_df(
                "SELECT scoring_event_id, customer_id, risk_score, "
                "decision_band, scored_at "
                "FROM scoring_events ORDER BY scored_at DESC LIMIT 20"
            )
            if not recent_df.empty:
                st.dataframe(recent_df, width="stretch", hide_index=True)

        ch.close()
    except Exception as e:
        st.warning(f"ClickHouse not available: {e}")
        st.info("Start ClickHouse and score some customers to see analytics.")
