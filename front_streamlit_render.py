import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

# ─── Config ───────────────────────────────────────────────────────────────────
# On Render: set the BACKEND_URL environment variable to your backend service URL
# e.g.  https://your-backend-name.onrender.com
# Locally: defaults to http://127.0.0.1:8000
API = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="Pro Stock Suite", layout="wide", page_icon="💹")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { height: 46px; border-radius: 5px; }
.stTabs [aria-selected="true"] { background-color: #4CAF50; color: white; }
.winner-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; padding: 12px 18px; border-radius: 10px;
    font-size: 1.05rem; font-weight: bold; margin: 8px 0;
}
.warn-box {
    background: #fff3cd; border: 1px solid #ffc107;
    padding: 10px 14px; border-radius: 6px; font-size: 0.9rem;
}
.regime-card {
    border-radius: 10px; padding: 14px 18px; margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.15);
}
.regime-high   { background: rgba(239, 83, 80, 0.12); border-left: 4px solid #EF5350; }
.regime-medium { background: rgba(255, 179, 0, 0.12); border-left: 4px solid #FFB300; }
.regime-low    { background: rgba(38, 166, 154, 0.12); border-left: 4px solid #26a69a; }
.rec-box {
    background: rgba(41, 98, 255, 0.10); border: 1px solid rgba(41,98,255,0.3);
    border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.88rem;
}
.metric-chip {
    display: inline-block; padding: 3px 10px;
    border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ─── Stock Data ───────────────────────────────────────────────────────────────
STOCKS = {
    "US Tech": {
        "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Tesla": "TSLA", "Amazon": "AMZN",
        "Google (Alphabet)": "GOOGL", "Meta": "META", "Netflix": "NFLX", "AMD": "AMD", "Intel": "INTC",
        "Salesforce": "CRM", "Adobe": "ADBE", "Oracle": "ORCL", "Cisco": "CSCO", "Qualcomm": "QCOM",
        "Broadcom": "AVGO", "Texas Instruments": "TXN", "IBM": "IBM", "Micron": "MU", "Applied Materials": "AMAT",
        "ServiceNow": "NOW", "Intuit": "INTU", "Uber": "UBER", "Airbnb": "ABNB", "Palantir": "PLTR",
        "Snowflake": "SNOW", "Block (Square)": "SQ", "PayPal": "PYPL", "Coinbase": "COIN", "Roblox": "RBLX",
        "Unity": "U", "Spotify": "SPOT", "Shopify": "SHOP", "Zoom": "ZM", "Twilio": "TWLO",
        "CrowdStrike": "CRWD", "Palo Alto": "PANW", "Fortinet": "FTNT", "Datadog": "DDOG", "Atlassian": "TEAM"
    },
    "India NSE": {
        "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS", "Infosys": "INFY.NS",
        "ICICI Bank": "ICICIBANK.NS", "HUL": "HINDUNILVR.NS", "SBI": "SBIN.NS", "Bharti Airtel": "BHARTIARTL.NS",
        "ITC": "ITC.NS", "Kotak Bank": "KOTAKBANK.NS", "L&T": "LT.NS", "Axis Bank": "AXISBANK.NS",
        "Asian Paints": "ASIANPAINT.NS", "HCL Tech": "HCLTECH.NS", "Maruti Suzuki": "MARUTI.NS", "Titan": "TITAN.NS",
        "Bajaj Finance": "BAJFINANCE.NS", "Sun Pharma": "SUNPHARMA.NS", "Tata Motors": "TATAMOTORS.NS",
        "UltraTech": "ULTRACEMCO.NS", "Power Grid": "POWERGRID.NS", "NTPC": "NTPC.NS", "M&M": "M&M.NS",
        "Nestle India": "NESTLEIND.NS", "JSW Steel": "JSWSTEEL.NS", "Tata Steel": "TATASTEEL.NS", "Grasim": "GRASIM.NS",
        "Tech Mahindra": "TECHM.NS", "Adani Ent": "ADANIENT.NS", "Adani Ports": "ADANIPORTS.NS", "Wipro": "WIPRO.NS",
        "Hindalco": "HINDALCO.NS", "Cipla": "CIPLA.NS", "SBI Life": "SBILIFE.NS", "Dr Reddys": "DRREDDY.NS",
        "Britannia": "BRITANNIA.NS", "Coal India": "COALINDIA.NS", "Tataconsumer": "TATACONSUM.NS",
        "Eicher Motors": "EICHERMOT.NS"
    },
    "Energy": {
        "Exxon Mobil": "XOM", "Chevron": "CVX", "Shell": "SHEL", "TotalEnergies": "TTE", "BP": "BP",
        "ConocoPhillips": "COP", "Schlumberger": "SLB", "EOG Resources": "EOG", "Pioneer Natural": "PXD",
        "Marathon Petroleum": "MPC", "Valero": "VLO", "Phillips 66": "PSX", "Occidental": "OXY", "Hess": "HES",
        "Kinder Morgan": "KMI", "Williams Co": "WMB", "ONEOK": "OKE", "TC Energy": "TRP", "Enbridge": "ENB",
        "Canadian Natural": "CNQ", "Suncor": "SU", "Cenovus": "CVE", "Imperial Oil": "IMO", "Halliburton": "HAL",
        "Baker Hughes": "BKR", "Devon Energy": "DVN", "Diamondback": "FANG", "Coterra": "CTRA", "Marathon Oil": "MRO",
        "APA Corp": "APA", "EQT Corp": "EQT", "Targa Resources": "TRGP", "Cheniere": "LNG", "NextEra Energy": "NEE",
        "Duke Energy": "DUK", "Southern Co": "SO", "Dominion": "D", "Exelon": "EXC", "American Electric": "AEP"
    },
    "Crypto": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD", "BNB": "BNB-USD", "XRP": "XRP-USD",
        "Cardano": "ADA-USD", "Dogecoin": "DOGE-USD", "Avalanche": "AVAX-USD", "Shiba Inu": "SHIB-USD",
        "Polkadot": "DOT-USD", "Tron": "TRX-USD", "Chainlink": "LINK-USD", "Polygon": "MATIC-USD",
        "Litecoin": "LTC-USD", "Bitcoin Cash": "BCH-USD", "Uniswap": "UNI-USD", "Cosmos": "ATOM-USD",
        "Stellar": "XLM-USD", "Ethereum Classic": "ETC-USD", "Filecoin": "FIL-USD",
        "Internet Computer": "ICP-USD", "Hedera": "HBAR-USD", "VeChain": "VET-USD", "Near": "NEAR-USD",
        "Aptos": "APT-USD", "Algorand": "ALGO-USD", "Quant": "QNT-USD", "Aave": "AAVE-USD",
        "Fantoms": "FTM-USD", "The Graph": "GRT-USD", "Sandbox": "SAND-USD", "Decentraland": "MANA-USD",
        "EOS": "EOS-USD", "Tezos": "XTZ-USD", "Flow": "FLOW-USD", "Axie Infinity": "AXS-USD", "Theta": "THETA-USD"
    },
}

ALL_TICKERS = {}
for _cat, _items in STOCKS.items():
    for _name, _sym in _items.items():
        ALL_TICKERS[f"{_name} ({_sym})"] = _sym

ALL_MODELS = [
    "XGBoost", "Random Forest", "Prophet", "ARIMA", "SARIMA",
    "Holt-Winters", "Ensemble", "Weighted Ensemble",
    "LSTM", "Bidirectional LSTM", "Adaptive LSTM",
]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🚀 Pro Stock Suite")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Forecasting Studio", "Model Battle Arena",
     "Growth Screener", "News Agent"],
)
st.sidebar.markdown("---")
st.sidebar.subheader("📖 Ticker Reference")
for cat, stocks in STOCKS.items():
    with st.sidebar.expander(f"📂 {cat}"):
        st.dataframe(
            pd.DataFrame(list(stocks.items()), columns=["Name", "Ticker"]),
            hide_index=True, use_container_width=True,
        )

# ─── Backend status
TF_OK       = True
BACKEND_OK  = True
GARCH_OK    = False
try:
    r = requests.get(f"{API}/", timeout=4)
    if r.status_code == 200:
        info = r.json()
        TF_OK    = info.get("tensorflow_available", True)
        GARCH_OK = info.get("garch_available", False)
    else:
        BACKEND_OK = False
except Exception:
    BACKEND_OK = False

if not BACKEND_OK:
    st.error(
        f"⚠️ Cannot reach backend at: **{API}**\n\n"
        "**Running on Render?** Go to your frontend service → Environment tab → "
        "add env var:  BACKEND_URL = https://your-backend-name.onrender.com\n\n"
        "**Running locally?** Start backend first:  uvicorn backend_fastapi_4:app --reload"
    )
    st.stop()

if not TF_OK:
    st.sidebar.markdown(
        '<div class="warn-box">⚠️ TensorFlow not installed — LSTM models unavailable.<br>'
        'Run: <code>pip install tensorflow</code></div>', unsafe_allow_html=True
    )

# ─── Helper: safe API call
def api_post(endpoint: str, body: dict, timeout: int = 60):
    try:
        r = requests.post(f"{API}{endpoint}", json=body, timeout=timeout)
        if r.status_code == 200:
            return r.json(), None
        else:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            return None, f"HTTP {r.status_code}: {detail}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Model may need more time."
    except Exception as ex:
        return None, str(ex)

def api_get(endpoint: str, timeout: int = 30):
    try:
        r = requests.get(f"{API}{endpoint}", timeout=timeout)
        if r.status_code == 200:
            return r.json(), None
        return None, f"HTTP {r.status_code}"
    except Exception as ex:
        return None, str(ex)


#dashboard
if page == "Dashboard":
    st.title("📊 Market Dashboard")
    col_l, col_r = st.columns([1, 4])

    with col_l:
        st.subheader("Settings")
        sel   = st.selectbox("Asset", list(ALL_TICKERS.keys()), key="d_sel")
        tick  = ALL_TICKERS[sel]
        s_dt  = st.date_input("Start", date.today() - timedelta(days=365), key="d_s")
        e_dt  = st.date_input("End",   date.today(),                       key="d_e")
        go_btn = st.button("Load", type="primary")
        if go_btn:
            st.session_state["dash_params"] = (tick, str(s_dt), str(e_dt))

    with col_r:
        params = st.session_state.get("dash_params")
        if not params:
            st.info("Select an asset and click Load.")
        else:
            tick, s, e = params
            data, err = api_post("/stock/history", {"ticker": tick,
                                                    "start_date": s, "end_date": e})
            if err:
                st.error(err)
            else:
                rows = data.get("data", [])
                if not rows:
                    st.warning("No rows returned.")
                else:
                    df = pd.DataFrame(rows)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                    df = df.dropna(subset=["Close"])

                    last  = float(df["Close"].iloc[-1])
                    prev  = float(df["Close"].iloc[-2]) if len(df) > 1 else last
                    delta = last - prev
                    pct   = (delta / prev * 100) if prev else 0

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Price",    f"{last:.2f}", f"{delta:+.2f} ({pct:+.2f}%)")
                    c2.metric("Volume",   f"{int(df['Volume'].iloc[-1]):,}" if "Volume" in df.columns else "—")
                    c3.metric("52W High", f"{df['High'].max():.2f}" if "High" in df.columns else "—")

                    t1, t2, t3 = st.tabs(["Line", "Candlestick", "Indicators"])

                    with t1:
                        fig = px.line(df, x="Date", y="Close",
                                      title=f"{tick} — Closing Price")
                        fig.update_traces(line_color="#2962FF", line_width=2)
                        st.plotly_chart(fig, use_container_width=True)

                    with t2:
                        if all(c in df.columns for c in ["Open","High","Low","Close"]):
                            fig = go.Figure(go.Candlestick(
                                x=df["Date"], open=df["Open"], high=df["High"],
                                low=df["Low"],   close=df["Close"],
                            ))
                            fig.update_layout(title=f"{tick} — Candlestick")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("OHLC columns not available.")

                    with t3:
                        ind, err2 = api_post("/stock/indicators",
                                             {"ticker": tick, "start_date": s, "end_date": e})
                        if err2:
                            st.warning(f"Indicators unavailable: {err2}")
                        else:
                            d_idx = ind.get("dates", [])

                            fig_bb = go.Figure()
                            fig_bb.add_trace(go.Scatter(x=d_idx, y=ind.get("bb_upper"),
                                name="BB Upper", line=dict(color="#95E1D3", dash="dash", width=1)))
                            fig_bb.add_trace(go.Scatter(x=d_idx, y=ind.get("bb_lower"),
                                name="BB Lower", fill="tonexty",
                                fillcolor="rgba(149,225,211,0.10)",
                                line=dict(color="#F38181", dash="dash", width=1)))
                            fig_bb.add_trace(go.Scatter(x=d_idx, y=ind.get("close"),
                                name="Price", line=dict(color="#2962FF", width=2)))
                            fig_bb.add_trace(go.Scatter(x=d_idx, y=ind.get("sma_20"),
                                name="SMA 20", line=dict(color="#FFE66D", width=1.5)))
                            fig_bb.add_trace(go.Scatter(x=d_idx, y=ind.get("sma_50"),
                                name="SMA 50", line=dict(color="#EAFFD0", width=1.5)))
                            fig_bb.update_layout(title="Bollinger Bands + MAs",
                                                 legend=dict(orientation="h"))
                            st.plotly_chart(fig_bb, use_container_width=True)

                            rr, mm = st.columns(2)
                            with rr:
                                fig_r = go.Figure()
                                fig_r.add_trace(go.Scatter(x=d_idx, y=ind.get("rsi"),
                                    name="RSI", line=dict(color="#FF6B6B", width=2)))
                                fig_r.add_hline(y=70, line=dict(color="red", dash="dot"),
                                                annotation_text="Overbought")
                                fig_r.add_hline(y=30, line=dict(color="green", dash="dot"),
                                                annotation_text="Oversold")
                                fig_r.update_layout(title="RSI (14)",
                                                    yaxis_range=[0, 100], height=260)
                                st.plotly_chart(fig_r, use_container_width=True)
                            with mm:
                                fig_m = go.Figure()
                                fig_m.add_trace(go.Scatter(x=d_idx, y=ind.get("macd"),
                                    name="MACD", line=dict(color="#4ECDC4", width=2)))
                                fig_m.add_trace(go.Scatter(x=d_idx, y=ind.get("macd_signal"),
                                    name="Signal", line=dict(color="#FFE66D", width=1.5)))
                                hist_vals = ind.get("macd_hist", [])
                                hist_cols = ["#26a69a" if (v or 0) >= 0 else "#ef5350"
                                             for v in hist_vals]
                                fig_m.add_trace(go.Bar(x=d_idx, y=hist_vals,
                                    name="Histogram", marker_color=hist_cols, opacity=0.6))
                                fig_m.update_layout(title="MACD", height=260)
                                st.plotly_chart(fig_m, use_container_width=True)

                    st.divider()
                    an, _ = api_post("/stock/analyze",
                                     {"ticker": tick, "start_date": s, "end_date": e},
                                     timeout=20)
                    if an:
                        a1, a2 = st.columns(2)
                        a1.info("Stationary: " + ("Yes" if an.get("is_stationary") else "No (Trending)"))
                        a2.info("Pattern: "    + ("Non-Linear" if an.get("is_nonlinear") else "Linear"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Forecasting Studio  (IMPROVED)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecasting Studio":
    st.title("🔮 Forecasting Studio")

    available_models = ALL_MODELS if TF_OK else [
        m for m in ALL_MODELS if m not in {"LSTM", "Bidirectional LSTM", "Adaptive LSTM"}
    ]

    col_l, col_r = st.columns([1, 3])

    # ─────────────────────── LEFT COLUMN ──────────────────────────────────────
    with col_l:
        st.subheader("Settings")
        sel  = st.selectbox("Asset", list(ALL_TICKERS.keys()), key="fc_sel")
        tick = ALL_TICKERS[sel]

        # ── Step 1: Analyze the stock profile ─────────────────────────────────
        st.markdown("**Step 1 — Analyze Stock**")
        if st.button("🔍 Analyze Stock Profile", use_container_width=True, key="btn_analyze"):
            with st.spinner(f"Running diagnostic on {tick}…"):
                regime_data, regime_err = api_get(f"/stock/regime/{tick}", timeout=40)
            if regime_err:
                st.error(f"Analysis failed: {regime_err}")
            elif regime_data:
                st.session_state["regime_data"]   = regime_data
                st.session_state["regime_ticker"] = tick
                # Auto-set model to recommended
                rec = regime_data.get("recommended_model", "")
                if rec and rec in available_models:
                    st.session_state["fc_auto_model"] = rec

        # ── Showing regime analysis card ──────────────────────────────────────────
        regime = st.session_state.get("regime_data")
        if regime and st.session_state.get("regime_ticker") == tick:
            vol_r = regime.get("regime", "medium")
            css_cls = {"high": "regime-high", "medium": "regime-medium", "low": "regime-low"}.get(vol_r, "regime-medium")
            vol_icons  = {"high": "🔴 High Volatility", "medium": "🟡 Medium Volatility", "low": "🟢 Low Volatility"}
            hurst_val  = regime.get("hurst_exponent", 0.5)
            hurst_label = "Trending" if hurst_val > 0.55 else ("Mean-Rev." if hurst_val < 0.45 else "Random Walk")
            adf_p = regime.get("adf_pvalue", 1.0)

            st.markdown(f"""
            <div class="regime-card {css_cls}">
            <b>{vol_icons.get(vol_r, vol_r)}</b><br>
            Ann. Volatility: <b>{regime.get('ann_volatility', 0)*100:.1f}%</b><br>
            Hurst: <b>{hurst_val:.3f}</b> — {hurst_label}<br>
            ADF p-val: <b>{adf_p:.3f}</b> {'(Stationary)' if adf_p < 0.05 else '(Non-Stationary)'}<br>
            ARCH Effect: <b>{'⚡ Yes — volatility clusters' if regime.get('arch_effect') else 'No'}</b><br>
            Seasonality: <b>{'📅 ' + str(regime.get('seasonal_period',0)) + '-day (' + str(round(regime.get('seasonal_strength',0),2)) + ')' if regime.get('seasonal') else 'Not detected'}</b><br>
            Trend: <b>{regime.get('trend_direction','sideways').title()}</b> {'(significant)' if regime.get('trending') else ''}
            </div>
            """, unsafe_allow_html=True)

            rec = regime.get("recommended_model", "")
            reasoning = regime.get("reasoning", "")
            if rec:
                st.markdown(f"""
                <div class="rec-box">
                💡 <b>Recommended Model: {rec}</b><br>
                <span style="font-size:0.82rem; opacity:0.85;">{reasoning}</span>
                </div>
                """, unsafe_allow_html=True)
                if rec in available_models:
                    if st.button(f"✨ Use {rec}", use_container_width=True, key="btn_use_rec"):
                        st.session_state["fc_auto_model"] = rec
                        st.rerun()

        st.divider()

        # ── Step 2: Choose model & days ────────────────────────────────────────
        st.markdown("**Step 2 — Configure Forecast**")

        auto_model_val = st.session_state.get("fc_auto_model", "XGBoost")
        model_idx = available_models.index(auto_model_val) if auto_model_val in available_models else 0
        model = st.selectbox("Model", available_models, index=model_idx, key="fc_model")

        days = st.slider("Forecast Days", 7, 365, 30)

        use_garch   = st.checkbox(
            "Confidence Bands (GARCH)" + (" ✓" if GARCH_OK else " — fallback to hist-vol"),
            value=True,
            help="90% confidence intervals using GARCH(1,1) if available, else historical volatility cone.",
        )
        auto_select = st.checkbox(
            "Auto-select best model",
            value=False,
            help="Ignores your model choice and uses the regime-recommended model.",
        )

        st.markdown("**Step 3 — Run**")
        if st.button("📅 Forecast 1 Year", use_container_width=True, key="btn_1yr"):
            st.session_state["fc_params"] = (tick, model, 252, use_garch, auto_select)
        if st.button("▶ Generate Forecast", type="primary", use_container_width=True, key="btn_gen"):
            st.session_state["fc_params"] = (tick, model, days, use_garch, auto_select)

        # ── Model tips ─────────────────────────────────────────────────────────
        tips = {
            "XGBoost":
                "**XGBoost** uses 25 technical indicators (RSI, MACD, Bollinger Bands, "
                "lags, volatility) as features. Best for trending, persistent markets (Hurst > 0.55).",
            "SARIMA":
                "**SARIMA**(p,d,q)(P,D,Q,s) adds seasonal autoregression on top of ARIMA. "
                "Ideal when ACF shows significant spikes at regular trading-day lags. "
                "Reference: Box & Jenkins (1976).",
            "Weighted Ensemble":
                "**Weighted Ensemble** assigns higher weight to models with lower "
                "recent CV error (walk-forward on last 30 days). "
                "More robust than equal-weight averaging. Reference: M5 competition (2022).",
            "Ensemble":
                "**Ensemble** averages XGBoost + Prophet + Holt-Winters. "
                "More stable than any one model.",
            "Adaptive LSTM":
                "**Adaptive LSTM** retrains on only the last 252 trading days, "
                "capturing recent market regime changes. Best for medium volatility.",
            "Bidirectional LSTM":
                "**BiLSTM** reads the series forward AND backward for richer patterns.",
            "LSTM":
                "**LSTM** best suits high-volatility, ARCH-affected stocks (NVDA, crypto). "
                "Reference: Selvin et al. (2017 IEEE DSAA).",
            "Prophet":
                "**Prophet** decomposes trend + weekly + annual seasonality + holiday effects. "
                "Best when ACF seasonal strength > 0.40. Reference: Taylor & Letham (2018).",
            "ARIMA":
                "**ARIMA** is best for low-volatility, stationary (or easily differenced) series. "
                "Reference: Box & Jenkins (1976).",
        }
        if model in tips:
            st.info(tips[model])

    # ─────────────────────── RIGHT COLUMN ─────────────────────────────────────
    with col_r:
        p = st.session_state.get("fc_params")
        if not p:
            # Show placeholder with regime hint if analyzed
            regime = st.session_state.get("regime_data")
            if regime and st.session_state.get("regime_ticker") == tick:
                st.info(
                    f"Stock analyzed ✅ — Recommended model: **{regime.get('recommended_model','')}**. "
                    "Configure forecast and click **Generate Forecast**."
                )
            else:
                st.info("👈 Analyze the stock first, then generate a forecast.")
        else:
            t, m, d, garch_flag, auto_flag = p

            with st.spinner(f"Running **{m}** on {t} for {d} days…"):
                res, err = api_post(
                    "/forecast",
                    {"ticker": t, "model": m, "days": d,
                     "use_garch": garch_flag, "auto_model": auto_flag},
                    timeout=360,
                )

            if err:
                st.error(err)
            elif not res:
                st.error("Empty response from backend.")
            else:
                dates_out  = res.get("dates", [])
                values_out = res.get("values", [])
                ci_upper   = res.get("confidence_upper", [])
                ci_lower   = res.get("confidence_lower", [])
                regime_res = res.get("regime", {})
                model_used = res.get("model", m)

                if not dates_out or not values_out:
                    st.error("Backend returned empty dates or values.")
                elif len(dates_out) != len(values_out):
                    st.error(f"Length mismatch: {len(dates_out)} dates vs {len(values_out)} values.")
                else:
                    # ── Regime summary strip ───────────────────────────────────
                    if regime_res:
                        vol_r  = regime_res.get("regime", "medium")
                        vol_icons = {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low"}
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("Volatility Regime",
                                  vol_icons.get(vol_r, vol_r),
                                  f"{regime_res.get('ann_volatility',0)*100:.1f}% ann.")
                        r2.metric("Hurst Exponent",
                                  f"{regime_res.get('hurst_exponent',0.5):.3f}",
                                  "Trending" if regime_res.get("hurst_exponent",0.5) > 0.55
                                  else "Random Walk" if regime_res.get("hurst_exponent",0.5) < 0.45
                                  else "Mixed")
                        r3.metric("ARCH Effect",
                                  "⚡ Yes" if regime_res.get("arch_effect") else "None",
                                  "Vol clustering" if regime_res.get("arch_effect") else "Homoskedastic")
                        r4.metric("Seasonality",
                                  f"📅 {regime_res.get('seasonal_period',0)}d" if regime_res.get("seasonal") else "None",
                                  f"Strength {regime_res.get('seasonal_strength',0):.2f}" if regime_res.get("seasonal") else "—")

                        # Show model recommendation if auto-select switched model
                        rec_m = regime_res.get("recommended_model", "")
                        if auto_flag and rec_m and rec_m != m:
                            st.success(
                                f"🤖 Auto-selected **{model_used}** based on regime analysis. "
                                f"{regime_res.get('reasoning','')}"
                            )
                        elif rec_m and rec_m != model_used:
                            st.info(
                                f"💡 For this stock's regime, **{rec_m}** may perform better. "
                                f"{regime_res.get('reasoning','')}"
                            )

                    st.divider()

                    # ── Build forecast chart ───────────────────────────────────
                    df_pred = pd.DataFrame({
                        "Date":  pd.to_datetime(dates_out),
                        "Price": [float(v) for v in values_out],
                    })

                    fig = go.Figure()

                    # Historical context (last 1 year)
                    hist, _ = api_post("/stock/history", {
                        "ticker": t,
                        "start_date": str(date.today() - timedelta(days=365)),
                        "end_date":   str(date.today()),
                    })
                    hist_last_price = None
                    if hist:
                        hdf = pd.DataFrame(hist["data"])
                        hdf["Date"]  = pd.to_datetime(hdf["Date"])
                        hdf["Close"] = pd.to_numeric(hdf["Close"], errors="coerce")
                        hdf = hdf.dropna(subset=["Close"])
                        if not hdf.empty:
                            hist_last_price = float(hdf["Close"].iloc[-1])
                        fig.add_trace(go.Scatter(
                            x=hdf["Date"], y=hdf["Close"],
                            name="Historical", line=dict(color="#607D8B", width=1.5),
                        ))

                    # Confidence bands (plotted under the forecast line)
                    if garch_flag and ci_upper and ci_lower and len(ci_upper) == len(dates_out):
                        fig.add_trace(go.Scatter(
                            x=dates_out + dates_out[::-1],
                            y=ci_upper + ci_lower[::-1],
                            fill="toself",
                            fillcolor="rgba(255,107,53,0.10)",
                            line=dict(width=0),
                            name="90% Confidence Band",
                            showlegend=True,
                        ))
                        fig.add_trace(go.Scatter(
                            x=dates_out, y=ci_upper,
                            line=dict(color="rgba(255,107,53,0.35)", width=1, dash="dot"),
                            showlegend=False, name="CI Upper",
                        ))
                        fig.add_trace(go.Scatter(
                            x=dates_out, y=ci_lower,
                            line=dict(color="rgba(255,107,53,0.35)", width=1, dash="dot"),
                            showlegend=False, name="CI Lower",
                        ))

                    # Main forecast line
                    fig.add_trace(go.Scatter(
                        x=df_pred["Date"], y=df_pred["Price"],
                        name=f"{model_used} Forecast",
                        line=dict(color="#FF6B35", width=2.5, dash="dash"),
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_pred["Date"], y=df_pred["Price"],
                        mode="markers", name="Forecast Points",
                        marker=dict(color="#FF6B35", size=4), showlegend=False,
                    ))

                    ci_note = " + 90% GARCH CI" if (garch_flag and ci_upper) else " (no CI)"
                    fig.update_layout(
                        title=f"{model_used} — {d}-Day Forecast for {t}{ci_note}",
                        legend=dict(orientation="h"),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Summary metrics ────────────────────────────────────────
                    first_v = float(df_pred["Price"].iloc[0])
                    last_v  = float(df_pred["Price"].iloc[-1])
                    chg     = (last_v - first_v) / (abs(first_v) + 1e-9) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Forecast End Price", f"{last_v:.2f}")
                    m2.metric("Projected Change",   f"{chg:+.2f}%")

                    if hist_last_price:
                        vs_now = (last_v - hist_last_price) / (abs(hist_last_price) + 1e-9) * 100
                        m3.metric("vs Current Price", f"{vs_now:+.2f}%",
                                  f"Current: {hist_last_price:.2f}")
                    else:
                        m3.metric("vs Current Price", "—")

                    if garch_flag and ci_upper and ci_lower:
                        ci_width = float(ci_upper[-1]) - float(ci_lower[-1])
                        ci_pct   = ci_width / (abs(last_v) + 1e-9) * 100
                        m4.metric("CI Width at End", f"±{ci_pct/2:.1f}%",
                                  "GARCH 90% band" if GARCH_OK else "hist-vol band")
                    else:
                        m4.metric("CI Width at End", "—")

                    # ── Confidence band table (expandable) ───────────────────
                    if garch_flag and ci_upper and ci_lower:
                        with st.expander("📊 Confidence Interval Table"):
                            ci_df = pd.DataFrame({
                                "Date":        dates_out,
                                "Forecast":    [round(float(v), 2) for v in values_out],
                                "Lower 90%":   [round(float(v), 2) for v in ci_lower],
                                "Upper 90%":   [round(float(v), 2) for v in ci_upper],
                                "Band Width":  [
                                    round(float(u) - float(l), 2)
                                    for u, l in zip(ci_upper, ci_lower)
                                ],
                            })
                            st.dataframe(ci_df, use_container_width=True, hide_index=True)

                    # ── Regime diagnostics (expandable) ──────────────────────
                    if regime_res:
                        with st.expander("🔬 Regime Diagnostics Detail"):
                            diag_cols = st.columns(2)
                            with diag_cols[0]:
                                st.markdown("**Volatility**")
                                st.write(f"Annualized: `{regime_res.get('ann_volatility',0)*100:.2f}%`")
                                st.write(f"Regime: `{regime_res.get('regime','').upper()}`")
                                st.write(f"ARCH Effect: `{regime_res.get('arch_effect')}`")
                                st.markdown("**Trend**")
                                st.write(f"Trending: `{regime_res.get('trending')}`")
                                st.write(f"Direction: `{regime_res.get('trend_direction','').upper()}`")
                            with diag_cols[1]:
                                st.markdown("**Time-Series Properties**")
                                st.write(f"ADF p-value: `{regime_res.get('adf_pvalue',1.0):.4f}` "
                                         f"({'Stationary' if regime_res.get('adf_pvalue',1)<0.05 else 'Non-Stationary'})")
                                st.write(f"Hurst Exponent: `{regime_res.get('hurst_exponent',0.5):.4f}`")
                                st.markdown("**Seasonality**")
                                st.write(f"Detected: `{regime_res.get('seasonal')}`")
                                if regime_res.get("seasonal"):
                                    st.write(f"Period: `{regime_res.get('seasonal_period',0)} trading days`")
                                    st.write(f"Strength: `{regime_res.get('seasonal_strength',0):.3f}`")
                            st.markdown("**Algorithm Recommendation**")
                            st.markdown(f"> {regime_res.get('reasoning','')}")
                            st.caption(
                                "References: Selvin et al. (2017 IEEE DSAA) · Taylor & Letham (2018) · "
                                "Bollerslev (1986) · Peters (1994) Fractal Market Analysis · "
                                "Vijh et al. (2021) · Hyndman & Athanasopoulos (2021)"
                            )

                    # ── Forecast table (expandable) ───────────────────────────
                    with st.expander("View forecast table"):
                        st.dataframe(df_pred, use_container_width=True)


#model
elif page == "Model Battle Arena":
    st.title("⚔️ Model Battle Arena")
    st.markdown("All models train on 80% of your date range and predict the rest. "
                "**Use 2+ years** for reliable results.")

    col_l, col_r = st.columns([1, 3])

    with col_l:
        st.subheader("Settings")
        sel2   = st.selectbox("Asset", list(ALL_TICKERS.keys()), key="ev_sel")
        e_tick = ALL_TICKERS[sel2]
        e_s    = st.date_input("Start", date.today() - timedelta(days=730), key="ev_s")
        e_e    = st.date_input("End",   date.today(),                       key="ev_e")
        st.caption("LSTM models can take 2–5 min. All others finish in <30 s.")

        if st.button("⚔️ Run Battle", type="primary", use_container_width=True):
            with st.spinner("Evaluating all models… please wait"):
                res, err = api_post("/forecast/evaluate", {
                    "ticker": e_tick,
                    "start_date": str(e_s),
                    "end_date": str(e_e),
                }, timeout=600)
            if err:
                st.error(err)
            elif not res:
                st.error("No results returned. Try a wider date range.")
            else:
                required = {"model", "rmse", "mape", "r2",
                            "dir_acc", "y_true", "y_pred", "dates"}
                valid = [r for r in res if required.issubset(r.keys())]
                if not valid:
                    st.error(
                        f"Backend returned {len(res)} result(s) but none have the "
                        "required keys. Ensure the backend is the latest version."
                    )
                else:
                    st.session_state["eval_res"]    = valid
                    st.session_state["eval_ticker"] = e_tick
                    if len(valid) < len(res):
                        st.warning(f"{len(res)-len(valid)} model(s) had incomplete data and were skipped.")
                    st.success(f"✅ Evaluated {len(valid)} models successfully.")

    with col_r:
        results = st.session_state.get("eval_res")
        if not results:
            st.info("👈 Run the battle to see results.")
        else:
            label = st.session_state.get("eval_ticker", "")

            rows = []
            for r in results:
                rows.append({
                    "Model":    r["model"],
                    "MAPE (%)": float(r.get("mape", 999)),
                    "R²":       float(r.get("r2", -999)),
                    "Dir Acc":  float(r.get("dir_acc", 50)),
                    "RMSE":     float(r.get("rmse", 999)),
                })
            mdf = pd.DataFrame(rows)

            best_row = mdf.loc[mdf["MAPE (%)"].idxmin()]
            st.markdown(
                f'<div class="winner-box">🏆 Best: {best_row["Model"]} &nbsp;|&nbsp;'
                f' MAPE {best_row["MAPE (%)"]:.2f}% &nbsp;|&nbsp;'
                f' Dir Acc {best_row["Dir Acc"]:.1f}%</div>',
                unsafe_allow_html=True,
            )

            tabs = st.tabs([
                "📊 MAPE & R²",
                "🎯 Directional Accuracy",
                "📈 Actual vs Predicted",
                "🕸 Radar",
            ])

            with tabs[0]:
                st.caption("MAPE: lower = better. R²: closer to 1.0 = better.")
                dm = mdf.sort_values("MAPE (%)")
                fig1 = go.Figure(go.Bar(
                    x=dm["Model"], y=dm["MAPE (%)"],
                    text=[f"{v:.2f}%" for v in dm["MAPE (%)"]],
                    textposition="outside",
                    marker_color=[
                        "#26a69a" if m == best_row["Model"] else "#EF5350"
                        for m in dm["Model"]
                    ],
                ))
                fig1.update_layout(
                    title=f"MAPE — {label}",
                    yaxis_title="MAPE (%)",
                    yaxis=dict(range=[0, dm["MAPE (%)"].max() * 1.3]),
                    showlegend=False,
                )
                st.plotly_chart(fig1, use_container_width=True)

                dr = mdf.sort_values("R²", ascending=False)
                fig2 = go.Figure(go.Bar(
                    x=dr["Model"], y=dr["R²"],
                    text=[f"{v:.3f}" for v in dr["R²"]],
                    textposition="outside",
                    marker_color=[
                        "#26a69a" if v >= 0.7 else "#FFB300" if v >= 0.4 else "#EF5350"
                        for v in dr["R²"]
                    ],
                ))
                fig2.add_hline(y=0, line=dict(color="gray", dash="dot"))
                fig2.update_layout(title="R² Score", yaxis_title="R²", showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            with tabs[1]:
                st.caption("50% = coin flip. Above 55% means the model has real directional skill.")
                dd = mdf.sort_values("Dir Acc", ascending=False)
                fig3 = go.Figure(go.Bar(
                    x=dd["Model"], y=dd["Dir Acc"],
                    text=[f"{v:.1f}%" for v in dd["Dir Acc"]],
                    textposition="outside",
                    marker_color=[
                        "#26a69a" if v >= 60 else "#FFB300" if v >= 52 else "#EF5350"
                        for v in dd["Dir Acc"]
                    ],
                ))
                fig3.add_hline(y=50, line=dict(color="gray", dash="dot"),
                               annotation_text="Random (50%)",
                               annotation_position="bottom right")
                fig3.update_layout(
                    title=f"Directional Accuracy — {label}",
                    yaxis_title="DA (%)",
                    yaxis=dict(range=[0, 110]),
                    showlegend=False,
                )
                st.plotly_chart(fig3, use_container_width=True)

                dr2 = mdf.sort_values("RMSE")
                fig4 = go.Figure(go.Bar(
                    x=dr2["Model"], y=dr2["RMSE"],
                    text=[f"{v:.2f}" for v in dr2["RMSE"]],
                    textposition="outside",
                    marker_color=[
                        "#26a69a" if m == best_row["Model"] else "#90A4AE"
                        for m in dr2["Model"]
                    ],
                ))
                fig4.update_layout(title="RMSE (lower = better)",
                                   yaxis_title="RMSE", showlegend=False)
                st.plotly_chart(fig4, use_container_width=True)

            with tabs[2]:
                model_names = [r["model"] for r in results]
                chosen = st.selectbox("Pick a model:", model_names, key="avp_pick")
                rd = next((r for r in results if r["model"] == chosen), None)

                if rd is None:
                    st.warning("No data for this model.")
                else:
                    yt  = rd.get("y_true", [])
                    yp  = rd.get("y_pred", [])
                    dts = rd.get("dates", [])
                    k   = min(len(yt), len(yp), len(dts))

                    if k < 2:
                        st.warning("Not enough data points to plot.")
                    else:
                        yt, yp, dts = yt[:k], yp[:k], dts[:k]
                        row_m = mdf[mdf["Model"] == chosen].iloc[0]

                        fig5 = go.Figure()
                        fig5.add_trace(go.Scatter(
                            x=dts, y=yt, name="Actual",
                            line=dict(color="#2962FF", width=2),
                        ))
                        fig5.add_trace(go.Scatter(
                            x=dts, y=yp, name=f"{chosen} Predicted",
                            line=dict(color="#FF6B35", width=2, dash="dash"),
                        ))
                        upper = [float(a) * 1.10 for a in yt]
                        lower = [float(a) * 0.90 for a in yt]
                        fig5.add_trace(go.Scatter(
                            x=dts + dts[::-1], y=upper + lower[::-1],
                            fill="toself", fillcolor="rgba(41,98,255,0.07)",
                            line=dict(width=0), name="±10% Band",
                        ))
                        fig5.update_layout(
                            title=(f"{chosen}  —  MAPE {row_m['MAPE (%)']:.2f}%  |  "
                                   f"DA {row_m['Dir Acc']:.1f}%  |  R² {row_m['R²']:.3f}"),
                            hovermode="x unified",
                            legend=dict(orientation="h"),
                        )
                        st.plotly_chart(fig5, use_container_width=True)

                        resid = [float(p) - float(t) for t, p in zip(yt, yp)]
                        fig6 = go.Figure(go.Bar(
                            x=dts, y=resid,
                            marker_color=["#EF5350" if v > 0 else "#26a69a" for v in resid],
                        ))
                        fig6.add_hline(y=0, line=dict(color="gray", dash="dot"))
                        fig6.update_layout(title="Residuals (Predicted − Actual)", height=220)
                        st.plotly_chart(fig6, use_container_width=True)

            with tabs[3]:
                st.caption("Normalised 0–1 across 4 metrics. Bigger area = better overall.")

                def _hex_rgba(hex_str: str, alpha: float = 0.20) -> str:
                    h = hex_str.lstrip("#")
                    if not h or len(h) not in (6, 3):
                        return f"rgba(100,100,200,{alpha})"
                    if len(h) == 3:
                        h = "".join(c * 2 for c in h)
                    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    return f"rgba({r},{g},{b},{alpha})"

                def norm_lo(a): mn,mx=a.min(),a.max(); return 1-(a-mn)/(mx-mn+1e-9)
                def norm_hi(a): mn,mx=a.min(),a.max(); return  (a-mn)/(mx-mn+1e-9)

                n_mape = norm_lo(mdf["MAPE (%)"].values)
                n_r2   = norm_hi(mdf["R²"].values.clip(0))
                n_da   = norm_hi(mdf["Dir Acc"].values)
                n_rmse = norm_lo(mdf["RMSE"].values)
                cats   = ["MAPE", "R²", "Dir Acc", "RMSE"]
                pal    = px.colors.qualitative.Plotly

                fig7 = go.Figure()
                for i, row in mdf.reset_index(drop=True).iterrows():
                    v = [n_mape[i], n_r2[i], n_da[i], n_rmse[i]]
                    v = v + [v[0]]
                    c = cats + [cats[0]]
                    hex_color = pal[i % len(pal)]
                    fig7.add_trace(go.Scatterpolar(
                        r=v, theta=c, fill="toself",
                        fillcolor=_hex_rgba(hex_color, 0.20),
                        line=dict(color=hex_color, width=2),
                        name=row["Model"],
                    ))
                fig7.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title=f"Radar — {label}", height=500,
                )
                st.plotly_chart(fig7, use_container_width=True)

                st.subheader("Full Table")
                st.dataframe(
                    mdf.sort_values("MAPE (%)").style
                    .highlight_min(subset=["MAPE (%)"], color="#c8f7c5")
                    .highlight_min(subset=["RMSE"],     color="#c8f7c5")
                    .highlight_max(subset=["R²"],       color="#c8f7c5")
                    .highlight_max(subset=["Dir Acc"],  color="#c8f7c5"),
                    use_container_width=True,
                )


#growth screener
elif page == "Growth Screener":
    st.title("⚡ Growth Screener")
    t1, t2, t3 = st.tabs(["Stocks Under $5", "Energy Stocks", "Crypto Top 20"])

    with t1:
        if st.button("Scan (<$5 Stocks)"):
            with st.spinner("Scraping…"):
                data, err = api_get("/scraper/stocks-under-5")
            if err:
                st.error(err)
            elif isinstance(data, list):
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.error(data.get("error", "Unknown error"))

    with t2:
        if st.button("Scan Energy"):
            with st.spinner("Scraping…"):
                data, err = api_get("/scraper/stocks-under-5?category=energy")
            if err:
                st.error(err)
            elif isinstance(data, list):
                st.dataframe(pd.DataFrame(data), use_container_width=True)
            else:
                st.error(data.get("error", "Unknown error"))

    with t3:
        if st.button("Fetch Crypto"):
            with st.spinner("Loading CoinGecko…"):
                data, err = api_get("/scraper/crypto")
            if err:
                st.error(err)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
                show = [c for c in ["symbol","name","current_price",
                                     "market_cap","price_change_percentage_24h"]
                        if c in df.columns]
                st.dataframe(df[show], use_container_width=True)
            else:
                st.error(data.get("error", "Unknown error") if isinstance(data, dict) else "Error")


# news agent tracker
elif page == "News Agent":
    st.title("📰 News Sentiment Agent")
    q = st.text_input("Ticker or company name", "Apple")

    if st.button("Analyse"):
        with st.spinner("Fetching and analysing news…"):
            data, err = api_get(f"/news/{q}", timeout=90)
        if err:
            st.error(err)
        elif not data:
            st.warning("No news found.")
        else:
            pos = sum(1 for x in data if x.get("sentiment") == "POSITIVE")
            neg = sum(1 for x in data if x.get("sentiment") == "NEGATIVE")
            st.metric("Overall", "Bullish 🟢" if pos >= neg else "Bearish 🔴",
                      f"{pos} Positive / {neg} Negative")
            fig = go.Figure(go.Pie(
                labels=["Positive","Negative"], values=[max(pos,0), max(neg,0)],
                marker_colors=["#26a69a","#EF5350"], hole=0.4,
            ))
            fig.update_layout(height=260, title="Sentiment Split")
            st.plotly_chart(fig, use_container_width=True)
            for item in data:
                icon = "🟢" if item.get("sentiment") == "POSITIVE" else "🔴"
                with st.expander(f"{icon} {item.get('title','')}"):
                    st.write(f"**Score:** {item.get('score', 0):.4f}")
                    link = item.get("link","")
                    if link:
                        st.markdown(f"[Read Article]({link})")
