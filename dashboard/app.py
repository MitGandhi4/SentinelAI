import streamlit as st
import requests
import os
# ── 1. Page configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="SentinelAI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ── 2. API connection settings ─────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# ── 3. Page header ──────────────────────────────────────────────────
st.title("🛡️ SentinelAI — Autonomous SecOps Dashboard")
st.markdown("LLM-powered threat detection, enrichment, and incident reporting")

# ── 4. Sidebar — alert simulation controls ─────────────────────────
st.sidebar.header("Simulate an Alert")

attack_options = [
    "PortScan", "DDoS", "DoS Hulk", "SSH-Patator", "FTP-Patator",
    "Heartbleed", "Bot", "Web Attack XSS", "Web Attack Sql Injection",
    "Infiltration"
]

selected_attack = st.sidebar.selectbox("Attack Type", attack_options)

anomaly_score = st.sidebar.slider(
    "Anomaly Score",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.01
)

st.sidebar.markdown("---")
analyze_button = st.sidebar.button("🔍 Run Analysis", use_container_width=True)

# ── 5. Main content area ────────────────────────────────────────────
if analyze_button:

    # Check API health first
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"✅ Connected to SentinelAI API — {health['techniques_loaded']} ATT&CK techniques loaded")
    except Exception as e:
        st.error(f"❌ Cannot connect to API. Make sure it's running on {API_URL}")
        st.stop()

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Alert Summary")
        st.metric("Attack Type", selected_attack)
        st.metric("Anomaly Score", f"{anomaly_score:.2f}")

        if anomaly_score >= 0.8:
            st.error("🔴 HIGH severity alert")
        elif anomaly_score >= 0.5:
            st.warning("🟡 MEDIUM severity alert")
        else:
            st.info("🟢 LOW severity alert")

    # ── 6. Generate the incident report ────────────────────────────
    with st.spinner("🤖 SentinelAI agent is analyzing the threat..."):
        try:
            report_response = requests.post(
                f"{API_URL}/report",
                json={
                    "attack_label": selected_attack,
                    "anomaly_score": anomaly_score
                },
                timeout=60
            )
            report_data = report_response.json()
        except Exception as e:
            st.error(f"Failed to generate report: {str(e)}")
            st.stop()

    with col2:
        st.subheader("🎯 MITRE ATT&CK Mapping")
        report_json = report_data.get("report_json", {})
        st.write(f"**Technique:** {report_json.get('mitre_technique', 'N/A')}")
        st.write(f"**Tactic:** {report_json.get('tactic', 'N/A')}")
        st.write(f"**Confidence:** {report_json.get('confidence', 'N/A')}")

    st.markdown("---")

    # ── 7. Display the full incident report ─────────────────────────
    st.subheader("📄 Generated Incident Report")
    st.markdown(report_data.get("report_markdown", "No report generated"))

else:
    st.info("👈 Select an attack type and click 'Run Analysis' in the sidebar to begin")