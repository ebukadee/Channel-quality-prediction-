
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="IQP • Intelligent Channel Quality Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
:root { --bg:#07111f; --panel:#0d1b2a; --line:#1c3852; --text:#e8f1f8; --muted:#91a8bb; }
.stApp { background:
    radial-gradient(circle at 85% 10%, rgba(0,229,255,.10), transparent 28%),
    radial-gradient(circle at 10% 90%, rgba(123,92,255,.10), transparent 30%),
    #07111f; color:var(--text); }
.block-container { padding-top: 2rem; max-width: 1250px; }
.hero { padding: 24px 28px; border:1px solid var(--line); border-radius:22px;
    background:linear-gradient(135deg, rgba(13,27,42,.96), rgba(10,24,40,.76));
    box-shadow:0 15px 45px rgba(0,0,0,.22); margin-bottom:20px; }
.hero h1 { margin:0; font-size:2.15rem; letter-spacing:-.03em; }
.hero p { color:var(--muted); margin:.45rem 0 0; }
.card { border:1px solid var(--line); border-radius:18px; padding:18px;
    background:rgba(13,27,42,.78); min-height:120px; }
.metric-label { color:var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; }
.metric-value { font-size:1.9rem; font-weight:700; margin-top:6px; }
.status-good { color:#5ee6a8; }
.status-mid { color:#ffd166; }
.status-bad { color:#ff6b7a; }
.small { color:var(--muted); font-size:.85rem; }
div[data-testid="stMetric"] { background:rgba(13,27,42,.78); border:1px solid var(--line);
    padding:12px; border-radius:16px; }
</style>
""", unsafe_allow_html=True)

FEATURES = ["snr_db", "rssi_dbm", "interference_db", "latency_ms", "bandwidth_mhz"]

@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 1800
    snr = rng.uniform(0, 35, n)
    rssi = rng.uniform(-110, -40, n)
    interference = rng.uniform(-105, -45, n)
    latency = rng.uniform(5, 180, n)
    bandwidth = rng.uniform(1, 100, n)

    # Synthetic but physically sensible quality relationship for a classroom prototype.
    quality = (
        42
        + 1.55 * snr
        + 0.25 * (rssi + 110)
        - 0.42 * (interference + 105)
        - 0.13 * latency
        + 0.10 * bandwidth
        + rng.normal(0, 5, n)
    )
    quality = np.clip(quality, 0, 100)

    X = pd.DataFrame({
        "snr_db": snr, "rssi_dbm": rssi, "interference_db": interference,
        "latency_ms": latency, "bandwidth_mhz": bandwidth
    })
    y = quality
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=140, max_depth=9, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, mean_absolute_error(y_test, pred), r2_score(y_test, pred)

model, mae, r2 = train_model()

def classify(score):
    if score >= 75:
        return "EXCELLENT", "status-good"
    if score >= 50:
        return "FAIR", "status-mid"
    return "POOR", "status-bad"

def advice(snr, rssi, interference, latency, bandwidth):
    items = []
    if snr < 12: items.append("Improve signal-to-noise ratio: move closer to the access point or improve antenna placement.")
    if rssi < -80: items.append("Weak received signal: consider reducing distance, obstacles, or transmit-power constraints.")
    if interference > -65: items.append("High interference detected: change channel/frequency or reduce nearby interferers.")
    if latency > 100: items.append("High latency: inspect congestion, routing, or queueing.")
    if bandwidth < 10: items.append("Low available bandwidth: reduce load or increase channel capacity.")
    return items or ["Channel conditions look healthy. Continue monitoring for sudden changes."]

st.markdown("""
<div class="hero">
  <h1>📡 Intelligent Channel Quality Prediction</h1>
  <p>A lightweight machine-learning dashboard that estimates wireless channel quality from live-style measurements.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### INPUT SIGNALS")
    snr = st.slider("SNR (dB)", 0.0, 35.0, 22.0, 0.5,
                    help="Signal-to-noise ratio. Higher is generally better.")
    rssi = st.slider("RSSI (dBm)", -110.0, -40.0, -62.0, 1.0,
                     help="Received signal strength. Values closer to -40 dBm are stronger.")
    interference = st.slider("Interference (dBm)", -105.0, -45.0, -82.0, 1.0,
                             help="Approximate interfering signal level. More negative is generally better.")
    latency = st.slider("Latency (ms)", 5.0, 180.0, 35.0, 1.0)
    bandwidth = st.slider("Available bandwidth (MHz)", 1.0, 100.0, 40.0, 1.0)
    predict = st.button("RUN PREDICTION", type="primary", use_container_width=True)

features = pd.DataFrame([{
    "snr_db": snr, "rssi_dbm": rssi, "interference_db": interference,
    "latency_ms": latency, "bandwidth_mhz": bandwidth
}])

score = float(np.clip(model.predict(features)[0], 0, 100))
label, css = classify(score)

st.caption("Prototype mode • Synthetic training data • Prediction updates automatically")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="card"><div class="metric-label">Predicted Quality</div><div class="metric-value {css}">{score:.1f}/100</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card"><div class="metric-label">Channel State</div><div class="metric-value {css}">{label}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="card"><div class="metric-label">Model R² / MAE</div><div class="metric-value">{r2:.2f} / {mae:.2f}</div></div>', unsafe_allow_html=True)

st.markdown("### Signal profile")
left, right = st.columns([1.25, 1])
with left:
    chart_df = pd.DataFrame({
        "Metric": ["SNR", "RSSI", "Interference", "Latency", "Bandwidth"],
        "Normalized level": [
            snr / 35 * 100,
            (rssi + 110) / 70 * 100,
            (-interference - 45) / 60 * 100,
            (180 - latency) / 175 * 100,
            bandwidth
        ]
    }).set_index("Metric")
    st.bar_chart(chart_df)
with right:
    st.markdown("#### Recommendation engine")
    for item in advice(snr, rssi, interference, latency, bandwidth):
        st.info(item)

st.markdown("### Current input packet")
st.dataframe(features.rename(columns={
    "snr_db":"SNR (dB)", "rssi_dbm":"RSSI (dBm)",
    "interference_db":"Interference (dBm)", "latency_ms":"Latency (ms)",
    "bandwidth_mhz":"Bandwidth (MHz)"
}), use_container_width=True, hide_index=True)

with st.expander("About the model"):
    st.write(
        "The demonstration uses a Random Forest regressor trained on generated wireless-channel measurements. "
        "The target is a 0–100 channel-quality score. The synthetic dataset is intentionally simple so the project "
        "can run without external datasets, sensors, APIs, or databases."
    )
    st.write("Important: this is an academic prototype, not a production radio-network optimization system.")
