#importing all Necessary libraries
import os
import datetime
import traceback
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import yfinance as yf
import requests as req
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
import feedparser
import transformers
from scipy.stats import linregress

# ── TensorFlow
TF_AVAILABLE = False
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM as KerasLSTM, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    print("[INFO] TensorFlow OK")
except Exception as e:
    print(f"[WARN] TensorFlow not available: {e}")

# ── GARCH model for volatality
GARCH_AVAILABLE = False
try:
    from arch import arch_model
    GARCH_AVAILABLE = True
    print("[INFO] arch/GARCH OK")
except Exception:
    print("[WARN] arch not installed — using historical-vol CI fallback. Run: pip install arch")

# ── pmdarima optional
PMDARIMA_AVAILABLE = False
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
    print("[INFO] pmdarima OK")
except Exception:
    print("[WARN] pmdarima not installed — SARIMA will use fixed order. Run: pip install pmdarima")

#statsmodels diagnostic
try:
    from statsmodels.stats.diagnostic import het_arch as arch_lm_test
    ARCH_TEST_AVAILABLE = True
except Exception:
    ARCH_TEST_AVAILABLE = False

app = FastAPI(title="Stock API")


# Pydantic
class StockReq(BaseModel):
    ticker: str
    start_date: str
    end_date: str

class EvalReq(BaseModel):
    ticker: str
    start_date: str
    end_date: str

class ForecastReq(BaseModel):
    ticker: str
    model: str
    days: int
    use_garch: bool = True      # compute GARCH confidence intervals
    auto_model: bool = False    # ignore model field and auto-select best


# Utilities
def pad_to(lst: list, n: int) -> list:
    lst = [float(x) for x in lst if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not lst:
        return [0.0] * n
    while len(lst) < n:
        lst.append(lst[-1])
    return lst[:n]


def get_close(data: pd.DataFrame) -> Optional[pd.Series]:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    for name in ["Close", "close", "Adj Close", "adj close"]:
        if name in data.columns:
            return data[name].squeeze()
    return None


def fetch_df(ticker: str, start, end) -> Optional[pd.DataFrame]:
    try:
        raw = yf.download(ticker, start=str(start), end=str(end), progress=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.reset_index(inplace=True)
        date_col = "Date" if "Date" in raw.columns else raw.columns[0]
        close_col = None
        for c in ["Close", "close", "Adj Close"]:
            if c in raw.columns:
                close_col = c
                break
        if close_col is None:
            return None
        df = raw[[date_col, close_col]].copy()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"]  = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        return df if len(df) > 0 else None
    except Exception as e:
        print(f"[fetch_df] {e}")
        return None


def fetch_raw(ticker: str, start, end) -> Optional[pd.DataFrame]:
    try:
        raw = yf.download(ticker, start=str(start), end=str(end), progress=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.reset_index(inplace=True)
        raw["Date"] = raw["Date"].astype(str) if "Date" in raw.columns else raw.iloc[:, 0].astype(str)
        raw = raw.ffill()
        return raw
    except Exception as e:
        print(f"[fetch_raw] {e}")
        return None


#  Technical Features for any stocks
FEAT_COLS = [
    "SMA10", "SMA20", "SMA50", "EMA12", "EMA26",
    "MACD", "MACD_sig", "MACD_hist",
    "RSI",
    "BB_upper", "BB_lower", "BB_width", "BB_pct",
    "Ret1", "Ret5", "Ret10", "Vol10", "Vol20",
    "Lag1", "Lag2", "Lag3", "Lag5", "Lag10",
    "PvSMA20",
]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().reset_index(drop=True)
    d["SMA10"]  = d["y"].rolling(10).mean()
    d["SMA20"]  = d["y"].rolling(20).mean()
    d["SMA50"]  = d["y"].rolling(50).mean()
    d["EMA12"]  = d["y"].ewm(span=12, adjust=False).mean()
    d["EMA26"]  = d["y"].ewm(span=26, adjust=False).mean()
    d["MACD"]      = d["EMA12"] - d["EMA26"]
    d["MACD_sig"]  = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["MACD_sig"]
    delta = d["y"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    bm = d["y"].rolling(20).mean()
    bs = d["y"].rolling(20).std()
    d["BB_upper"] = bm + 2 * bs
    d["BB_lower"] = bm - 2 * bs
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (bm + 1e-9)
    d["BB_pct"]   = (d["y"] - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"] + 1e-9)
    r = d["y"].pct_change()
    d["Ret1"]  = r
    d["Ret5"]  = d["y"].pct_change(5)
    d["Ret10"] = d["y"].pct_change(10)
    d["Vol10"] = r.rolling(10).std()
    d["Vol20"] = r.rolling(20).std()
    for k in [1, 2, 3, 5, 10]:
        d[f"Lag{k}"] = d["y"].shift(k)
    d["PvSMA20"] = d["y"] / (d["SMA20"] + 1e-9)
    return d.dropna().reset_index(drop=True)


# Metrics
def mape(yt, yp) -> float:
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    mask = np.abs(yt) > 1e-8
    if mask.sum() == 0:
        return 999.0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)

def da(yt, yp) -> float:
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    if len(yt) < 2:
        return 50.0
    return float(np.mean((np.diff(yt) > 0) == (np.diff(yp) > 0)) * 100)


#  Holt-Winters helper function
def hw_fit(series: pd.Series):
    n = len(series)
    for sp in [5, 12, 52]:
        if n >= sp * 2 + 2:
            try:
                return ExponentialSmoothing(series, trend="add", seasonal="add",
                                            seasonal_periods=sp).fit()
            except Exception:
                continue
    return ExponentialSmoothing(series, trend="add", seasonal=None).fit()


# Regime Detection & Diagnostics

def hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """
    Compute Hurst exponent via R/S analysis.
    H > 0.55  → persistent / trending (XGBoost / LSTM works well)
    H ~ 0.50  → random walk (ensemble / ARIMA)
    H < 0.45  → mean-reverting (ARIMA / Holt-Winters)
    Reference: Hurst (1951), Mandelbrot & Wallis (1969)
    """
    n = len(series)
    max_lag = min(max_lag, n // 2)
    if max_lag < 4:
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = np.subtract(series[lag:], series[:-lag])
        if np.std(diff) > 0:
            tau.append(np.std(diff))
        else:
            tau.append(1e-9)
    if len(tau) < 4:
        return 0.5
    try:
        reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(np.clip(reg[0], 0.0, 1.0))
    except Exception:
        return 0.5


def detect_arch_effect(returns: np.ndarray) -> bool:
    """
    ARCH-LM test for volatility clustering (heteroskedasticity).
    p < 0.05 → significant ARCH effect → GARCH modeling is appropriate.
    # this is to be noted that it is in Reference: Engle (1982)
    """
    if not ARCH_TEST_AVAILABLE or len(returns) < 30:
        return bool(np.std(returns[-60:]) / (np.std(returns) + 1e-9) > 1.3)
    try:
        _, lm_pval, _, _ = arch_lm_test(returns[-200:] if len(returns) > 200 else returns, nlags=10)
        return bool(lm_pval < 0.05)
    except Exception:
        return False


def detect_seasonality(series: np.ndarray) -> tuple:
    """
    Detect seasonality via ACF spikes at known trading-day lags.
    Returns (is_seasonal: bool, strength: float, period: int)
    Checks: weekly(5), monthly(21), quarterly(63) trading day lags.
    """
    if len(series) < 126:
        return False, 0.0, 0
    try:
        acf_vals = acf(series, nlags=70, fft=True)
        best_strength = 0.0
        best_period = 0
        for lag in [5, 21, 42, 63]:
            if lag < len(acf_vals):
                strength = abs(float(acf_vals[lag]))
                if strength > best_strength:
                    best_strength = strength
                    best_period = lag
        return best_strength > 0.25, round(best_strength, 4), best_period
    except Exception:
        return False, 0.0, 0


def detect_regime(df: pd.DataFrame) -> dict:
    """
    Comprehensive stock regime analysis. Returns:
    - volatility_regime: 'high' / 'medium' / 'low'
    - ann_volatility: annualized daily-return std
    - hurst_exponent: long-memory metric
    - arch_effect: bool — volatility clustering present
    - seasonal: bool — seasonal pattern detected
    - seasonal_strength: float
    - seasonal_period: int (trading days)
    - trending: bool
    - trend_direction: 'up' / 'down' / 'sideways'
    - recommended_model: best algorithm for this stock's characteristics
    - reasoning: plain-English explanation
    - adf_pvalue: stationarity test p-value

    Research basis:
    - Volatility regimes: Ang & Bekaert (2002), Hamilton (1989)
    - Hurst exponent: Peters (1994) "Fractal Market Analysis"
    - GARCH selection: Bollerslev (1986), Hansen & Lunde (2005)
    - Seasonal detection: Hyndman & Athanasopoulos (2021)
    """
    series = df["y"].values
    returns = pd.Series(series).pct_change().dropna().values

    if len(returns) < 20:
        return {
            "regime": "unknown", "ann_volatility": 0.0, "hurst_exponent": 0.5,
            "arch_effect": False, "seasonal": False, "seasonal_strength": 0.0,
            "seasonal_period": 0, "trending": False, "trend_direction": "sideways",
            "recommended_model": "XGBoost", "reasoning": "Insufficient data for analysis",
            "adf_pvalue": 1.0,
        }

    # 1. Annualized volatility
    window = min(252, len(returns))
    ann_vol = float(np.std(returns[-window:]) * np.sqrt(252))
    if ann_vol > 0.50:
        vol_regime = "high"
    elif ann_vol > 0.25:
        vol_regime = "medium"
    else:
        vol_regime = "low"

    # 2. Hurst exponent
    hurst = hurst_exponent(series[-500:] if len(series) > 500 else series)

    # 3. ARCH effect
    arch_effect = detect_arch_effect(returns)

    # 4. Seasonality
    is_seasonal, seasonal_strength, seasonal_period = detect_seasonality(series)

    # 5. Trend detection (90-day rolling OLS)
    trend_window = series[-90:] if len(series) >= 90 else series
    x_t = np.arange(len(trend_window))
    slope, _, r_val, p_val, _ = linregress(x_t, trend_window)
    trending = bool(p_val < 0.05 and abs(r_val) > 0.5)
    trend_direction = ("up" if slope > 0 else "down") if trending else "sideways"

    # 6. ADF test for stationarity
    adf_pval = 1.0
    try:
        adf_pval = float(adfuller(series[-500:] if len(series) > 500 else series)[1])
    except Exception:
        pass

    # ── Model recommendation logic ─────────────────────────────────────────────
    # Based on: Selvin et al. (2017), Lim et al. (2021 TFT paper),
    #           Taylor & Letham (2018 Prophet paper), Oreshkin et al. (2020 N-BEATS),
    #           Vijh et al. (2021) FAANG stock ML comparison
    if vol_regime == "high" and arch_effect:
        if TF_AVAILABLE:
            recommended = "LSTM"
            reasoning = (
                f"High volatility ({ann_vol:.0%} annualized) with ARCH volatility clustering. "
                "LSTM's gated memory handles nonlinear volatility spikes better than classical models. "
                "Reference: Selvin et al. (2017 IEEE DSAA)."
            )
        else:
            recommended = "XGBoost"
            reasoning = (
                f"High volatility ({ann_vol:.0%} ann.) with ARCH effects. "
                "XGBoost with technical features (RSI, Vol20, MACD) is the best non-DL option. "
                "Install TensorFlow for LSTM upgrade."
            )

    elif is_seasonal and seasonal_strength > 0.40:
        recommended = "Prophet"
        reasoning = (
            f"Strong seasonality detected at {seasonal_period}-day lag "
            f"(ACF strength {seasonal_strength:.2f}). "
            "Prophet decomposes trend + seasonal + holiday components explicitly. "
            "Reference: Taylor & Letham (2018, The American Statistician)."
        )

    elif is_seasonal and seasonal_strength > 0.25:
        recommended = "SARIMA"
        reasoning = (
            f"Moderate seasonality at {seasonal_period}-day lag "
            f"(ACF strength {seasonal_strength:.2f}). "
            "SARIMA(p,d,q)(P,D,Q,s) captures seasonal autoregression. "
            "Reference: Box & Jenkins (1976), Hyndman & Athanasopoulos (2021)."
        )

    elif vol_regime == "low" and trending:
        recommended = "ARIMA"
        reasoning = (
            f"Low volatility ({ann_vol:.0%} ann.) with a clear {trend_direction}trend "
            f"(R²={r_val**2:.2f}). Linear ARIMA captures this regime reliably. "
            "Reference: Box & Jenkins (1976)."
        )

    elif hurst > 0.58:
        recommended = "XGBoost"
        reasoning = (
            f"Strong trend persistence (Hurst={hurst:.3f} > 0.55). "
            "XGBoost with lag + technical features exploits momentum patterns. "
            "Reference: Vijh et al. (2021) — XGBoost+ARIMA outperformed standalone DL on MSFT."
        )

    elif hurst < 0.42:
        recommended = "Holt-Winters"
        reasoning = (
            f"Mean-reverting behavior (Hurst={hurst:.3f} < 0.45). "
            "Holt-Winters with additive smoothing is robust under mean-reversion. "
            "Price spikes are likely transient."
        )

    elif vol_regime == "medium" and TF_AVAILABLE:
        recommended = "Adaptive LSTM"
        reasoning = (
            f"Medium volatility ({ann_vol:.0%} ann.) with mixed signals. "
            "Adaptive LSTM retrains on the last 252 trading days to track regime changes. "
            "Reference: Nikou et al. (2020, Entropy journal)."
        )

    else:
        recommended = "Ensemble"
        reasoning = (
            "Mixed signals across all diagnostics. "
            "Weighted Ensemble (XGBoost + Prophet + Holt-Winters) averages their strengths "
            "and is the most robust choice when no single regime dominates."
        )

    return {
        "regime": vol_regime,
        "ann_volatility": round(ann_vol, 4),
        "hurst_exponent": round(hurst, 4),
        "arch_effect": arch_effect,
        "seasonal": is_seasonal,
        "seasonal_strength": seasonal_strength,
        "seasonal_period": seasonal_period,
        "trending": trending,
        "trend_direction": trend_direction,
        "recommended_model": recommended,
        "reasoning": reasoning,
        "adf_pvalue": round(adf_pval, 4),
        "data_points": len(df),
    }


#garch

def garch_confidence_bands(
    returns: np.ndarray,
    forecast_prices: list,
    horizon: int,
    confidence: float = 0.90,
) -> tuple:
    """
    Compute GARCH(1,1) confidence bands around a price forecast.

    Method:
    - Fit GARCH(1,1) on log returns (Bollerslev 1986)
    - Use conditional variance forecasts to build price-space CI
    - z = 1.645 for 90% CI, 1.96 for 95% CI

    Falls back to expanding historical-vol cone if arch not installed.
    Reference: Hansen & Lunde (2005, Journal of Applied Econometrics)
    """
    z = 1.645 if confidence == 0.90 else 1.96
    n = horizon
    fp = np.asarray(forecast_prices[:n], dtype=float)

    if GARCH_AVAILABLE and len(returns) >= 60:
        try:
            log_r = returns[-500:] if len(returns) > 500 else returns
            log_r = log_r * 100  # GARCH works better on percentage returns
            am = arch_model(log_r, vol="Garch", p=1, q=1, dist="Normal")
            res = am.fit(disp="off", show_warning=False)
            fcast = res.forecast(horizon=n, reindex=False)
            cond_var = fcast.variance.values[-1]            # shape (n,)
            cond_std = np.sqrt(cond_var) / 100              # back to decimal
            # cumulative std scales with sqrt(horizon step)
            cum_std = np.array([cond_std[:i+1].mean() * np.sqrt(i+1) for i in range(n)])
            upper = list(fp * np.exp(z * cum_std))
            lower = list(fp * np.exp(-z * cum_std))
            return [round(float(v), 4) for v in upper], [round(float(v), 4) for v in lower]
        except Exception as e:
            print(f"[GARCH] falling back to hist-vol CI: {e}")

    # Historical-vol fallback: expanding volatility cone
    hist_vol = float(np.std(returns[-60:] if len(returns) >= 60 else returns))
    cum_std = hist_vol * np.sqrt(np.arange(1, n + 1))
    upper = list(fp * (1.0 + z * cum_std))
    lower = list(fp * (1.0 - z * cum_std))
    return [round(float(v), 4) for v in upper], [round(float(v), 4) for v in lower]


# SARIMA forecast (a liear forecast prediction

def sarima_forecast(df: pd.DataFrame, n: int) -> list:
    """
    SARIMA forecast with automatic or fixed order.

    Uses pmdarima.auto_arima if available (stepwise AIC search).
    Falls back to SARIMAX(1,1,1)(1,1,0,s) with s auto-detected from
    ACF seasonal lags.

    Reference: Box & Jenkins (1976); Hyndman & Athanasopoulos (2021)
    """
    series = df["y"].values
    if len(series) < 60:
        return pad_to(list(series), n)

    # Detect best seasonal period
    _, _, sp = detect_seasonality(series)
    sp = sp if sp >= 5 else 5

    try:
        if PMDARIMA_AVAILABLE:
            model = pm.auto_arima(
                series, seasonal=True, m=sp,
                stepwise=True, suppress_warnings=True,
                error_action="ignore", max_p=3, max_q=3,
                max_P=1, max_Q=1, D=1, information_criterion="aic",
            )
            return pad_to(list(model.predict(n)), n)
        else:
            # Fixed sensible order — works for most financial series
            model = SARIMAX(
                series, order=(1, 1, 1),
                seasonal_order=(1, 1, 0, sp),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            return pad_to(list(res.forecast(steps=n)), n)
    except Exception as e:
        print(f"[SARIMA] falling back to ARIMA(5,1,0): {e}")
        try:
            fit = ARIMA(series, order=(5, 1, 0)).fit()
            return pad_to(list(fit.forecast(steps=n)), n)
        except Exception:
            return pad_to([float(series[-1])] * n, n)



#Weighted Ensemble
def weighted_ensemble_forecast(df: pd.DataFrame, n: int) -> list:
    """
    Weighted ensemble that assigns higher weight to models with lower
    recent cross-validation error (walk-forward on last 30 trading days).

    Models used: XGBoost, Prophet, Holt-Winters.
    Weights are inverse-RMSE normalized (softmin weighting).

    Reference: Makridakis et al. (2022, IJF M5 competition results)
    show weighted ensembles consistently outperform equal-weight averaging.
    """
    if len(df) < 80:
        # Not enough data for CV — fall back to equal-weight
        return _equal_ensemble_forecast(df, n)

    cv_horizon = min(30, len(df) // 5)
    split_idx  = len(df) - cv_horizon
    sub_models = ["XGBoost", "Prophet", "Holt-Winters"]

    weights = {}
    for sm in sub_models:
        try:
            yt_cv, yp_cv, _ = run_model_eval(df, sm, split_idx)
            k = min(len(yt_cv), len(yp_cv))
            if k < 2:
                weights[sm] = 1.0
            else:
                rmse_cv = float(np.sqrt(mean_squared_error(
                    np.array(yt_cv[:k]), np.array(yp_cv[:k])
                )))
                weights[sm] = 1.0 / (rmse_cv + 1e-6)
        except Exception as ex:
            print(f"[WeightedEnsemble CV/{sm}] {ex}")
            weights[sm] = 0.5   # partial credit

    total_w = sum(weights.values())
    if total_w == 0:
        return _equal_ensemble_forecast(df, n)

    # Generate forecasts and combine
    combined = np.zeros(n)
    used_w    = 0.0
    for sm in sub_models:
        w = weights.get(sm, 0.0) / total_w
        if w == 0:
            continue
        try:
            preds = run_forecast(df, sm, n)
            combined += w * np.array(pad_to(preds, n))
            used_w += w
        except Exception as ex:
            print(f"[WeightedEnsemble/{sm}] {ex}")

    if used_w == 0:
        return _equal_ensemble_forecast(df, n)

    combined /= used_w     # re-normalize if some models failed
    return [round(float(v), 4) for v in combined]


def _equal_ensemble_forecast(df: pd.DataFrame, n: int) -> list:
    bag = {}
    for sm in ["XGBoost", "Prophet", "Holt-Winters"]:
        try:
            bag[sm] = run_forecast(df, sm, n)
        except Exception as ex:
            print(f"[EqualEnsemble/{sm}] {ex}")
    if not bag:
        raise ValueError("all ensemble sub-models failed")
    arr = np.array(list(bag.values()), dtype=float)
    return list(arr.mean(axis=0))


# LSTM runner (only if TF available)
def lstm_forecast(series: np.ndarray, n_out: int, bidir: bool = False,
                  seq: int = 60, epochs: int = 60) -> list:
    if not TF_AVAILABLE:
        raise HTTPException(501, "TensorFlow not installed. Run: pip install tensorflow")

    from tensorflow.keras.optimizers import Adam

    series = np.asarray(series, float)
    n      = len(series)
    seq = max(20, min(seq, n // 3, 90))
    if n < seq + 10:
        return pad_to([float(series[-1])], n_out)

    sc      = MinMaxScaler()
    sc_data = sc.fit_transform(series.reshape(-1, 1))

    Xs, ys = [], []
    for i in range(seq, n):
        Xs.append(sc_data[i - seq:i, 0])
        ys.append(sc_data[i, 0])
    Xs = np.array(Xs).reshape(-1, seq, 1)
    ys = np.array(ys)

    batch = 16 if n < 400 else 32

    if bidir:
        mdl = Sequential([
            Bidirectional(KerasLSTM(128, return_sequences=True, input_shape=(seq, 1))),
            Dropout(0.25),
            Bidirectional(KerasLSTM(64, return_sequences=True)),
            Dropout(0.20),
            KerasLSTM(32, return_sequences=False),
            Dropout(0.15),
            Dense(16, activation="relu"),
            Dense(1),
        ])
    else:
        mdl = Sequential([
            KerasLSTM(128, return_sequences=True, input_shape=(seq, 1)),
            Dropout(0.25),
            KerasLSTM(64, return_sequences=True),
            Dropout(0.20),
            KerasLSTM(32, return_sequences=False),
            Dropout(0.15),
            Dense(16, activation="relu"),
            Dense(1),
        ])

    mdl.compile(optimizer=Adam(learning_rate=1e-3), loss="huber")

    from tensorflow.keras.callbacks import ReduceLROnPlateau
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", patience=5,
                          factor=0.5, min_lr=1e-6, verbose=0),
    ]

    mdl.fit(Xs, ys, epochs=epochs, batch_size=batch,
            validation_split=0.15, callbacks=callbacks, verbose=0)

    buf = sc_data[-seq:].copy()
    out = []
    for _ in range(n_out):
        p = float(mdl.predict(buf.reshape(1, seq, 1), verbose=0)[0][0])
        out.append(p)
        buf = np.vstack([buf[1:], [[p]]])

    inv = sc.inverse_transform(np.array(out).reshape(-1, 1)).flatten().tolist()
    return pad_to(inv, n_out)


# XGBoost model
def xgb_model():
    return xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_alpha=0.1, reg_lambda=1.0,
                             random_state=42, n_jobs=-1, verbosity=0)

def xgb_forecast(df: pd.DataFrame, n: int) -> list:
    feat = make_features(df)
    cols = [c for c in FEAT_COLS if c in feat.columns]
    X, y = feat[cols].values, feat["y"].values
    mdl = xgb_model()
    mdl.fit(X, y)
    lag_keys = sorted([int(c[3:]) for c in cols if c.startswith("Lag")])
    hist = list(df["y"].values[-(max(lag_keys, default=10) + 5):])
    row = feat[cols].iloc[-1].copy()
    preds = []
    for _ in range(n):
        p = float(mdl.predict(row.values.reshape(1, -1))[0])
        preds.append(p)
        hist.append(p)
        for k in lag_keys:
            col = f"Lag{k}"
            if col in row.index and len(hist) > k:
                row[col] = hist[-(k + 1)]
        if "PvSMA20" in row.index and len(hist) >= 20:
            row["PvSMA20"] = p / (float(np.mean(hist[-20:])) + 1e-9)
    return pad_to(preds, n)


# Core per-model prediction (train on [:split], predict [split:])
def run_model_eval(df: pd.DataFrame, name: str, split: int):
    yt   = df["y"].iloc[split:].values.astype(float)
    dts  = df["ds"].iloc[split:].dt.strftime("%Y-%m-%d").tolist()
    n    = len(yt)
    if n < 2:
        raise ValueError("too few test points")

    if name in ("XGBoost", "Random Forest"):
        feat = make_features(df)
        cols = [c for c in FEAT_COLS if c in feat.columns]
        adj = max(30, min(int(split * len(feat) / len(df)), len(feat) - 5))
        Xtr, Xte = feat[cols].values[:adj], feat[cols].values[adj:]
        ytr       = feat["y"].values[:adj]
        ytf       = feat["y"].values[adj:]
        dtf       = feat["ds"].dt.strftime("%Y-%m-%d").tolist()[adj:]
        if name == "XGBoost":
            mdl = xgb_model()
        else:
            mdl = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
        mdl.fit(Xtr, ytr)
        yp = mdl.predict(Xte)
        k  = min(len(ytf), len(yp), len(dtf))
        return list(ytf[:k].astype(float)), list(yp[:k].astype(float)), dtf[:k]

    elif name == "ARIMA":
        try:
            fit = ARIMA(df["y"].iloc[:split], order=(5, 1, 0)).fit()
        except Exception:
            fit = ARIMA(df["y"].iloc[:split], order=(1, 1, 1)).fit()
        yp = pad_to(list(fit.forecast(steps=n)), n)
        return list(yt), yp, dts

    elif name == "SARIMA":
        sub_df = df.iloc[:split].copy()
        yp = sarima_forecast(sub_df, n)
        return list(yt), pad_to(yp, n), dts

    elif name == "Prophet":
        m = Prophet(uncertainty_samples=0, daily_seasonality=False,
                    yearly_seasonality=True, weekly_seasonality=True)
        m.fit(df.iloc[:split][["ds", "y"]])
        fut = m.make_future_dataframe(periods=n + 30)
        fc  = m.predict(fut)
        raw = fc[fc["ds"] > df["ds"].iloc[split - 1]]["yhat"].tolist()
        yp  = pad_to(raw, n)
        return list(yt), yp, dts

    elif name == "Holt-Winters":
        fit = hw_fit(df["y"].iloc[:split])
        yp  = pad_to(list(fit.forecast(n)), n)
        return list(yt), yp, dts

    elif name == "LSTM":
        yp = lstm_forecast(df["y"].iloc[:split].values, n, bidir=False, epochs=60)
        return list(yt), pad_to(yp, n), dts

    elif name == "Bidirectional LSTM":
        yp = lstm_forecast(df["y"].iloc[:split].values, n, bidir=True, epochs=60)
        return list(yt), pad_to(yp, n), dts

    elif name == "Adaptive LSTM":
        window = min(504, split)
        seq    = max(20, min(60, window // 5))
        yp = lstm_forecast(df["y"].iloc[split - window:split].values, n,
                           bidir=False, seq=seq, epochs=60)
        return list(yt), pad_to(yp, n), dts

    elif name == "Ensemble":
        preds_bag = {}
        for sm in ["XGBoost", "Prophet", "Holt-Winters"]:
            try:
                _, yp, _ = run_model_eval(df, sm, split)
                preds_bag[sm] = pad_to(yp, n)
            except Exception as ex:
                print(f"[Ensemble/{sm}] {ex}")
        if len(preds_bag) < 1:
            raise ValueError("all ensemble sub-models failed")
        stk = np.array(list(preds_bag.values()), dtype=float)
        return list(yt), list(stk.mean(axis=0)), dts

    elif name == "Weighted Ensemble":
        # Use same 80/20 split logic
        sub_df = df.copy()
        preds_bag = {}
        for sm in ["XGBoost", "Prophet", "Holt-Winters"]:
            try:
                _, yp, _ = run_model_eval(sub_df, sm, split)
                preds_bag[sm] = pad_to(yp, n)
            except Exception as ex:
                print(f"[WEnsemble/{sm}] {ex}")
        if not preds_bag:
            raise ValueError("all weighted ensemble sub-models failed")
        stk = np.array(list(preds_bag.values()), dtype=float)
        return list(yt), list(stk.mean(axis=0)), dts

    raise ValueError(f"unknown model: {name}")


#─ /forecast helper
def run_forecast(df: pd.DataFrame, name: str, n: int) -> list:
    if name == "XGBoost":
        return xgb_forecast(df, n)

    elif name == "Random Forest":
        lag = 5
        dl = pd.DataFrame({"y": df["y"]})
        for i in range(1, lag + 1):
            dl[f"L{i}"] = dl["y"].shift(i)
        dl = dl.dropna()
        mdl = RandomForestRegressor(n_estimators=200, random_state=42)
        mdl.fit(dl.drop(columns=["y"]).values, dl["y"].values)
        hist = list(df["y"].tail(lag).values)
        out  = []
        for _ in range(n):
            p = float(mdl.predict(np.array(hist[-lag:]).reshape(1, -1))[0])
            out.append(p)
            hist.append(p)
        return pad_to(out, n)

    elif name == "Prophet":
        m = Prophet(uncertainty_samples=0, daily_seasonality=False,
                    yearly_seasonality=True, weekly_seasonality=True)
        m.fit(df)
        fut = m.make_future_dataframe(periods=n + 30)
        fc  = m.predict(fut)
        raw = fc[fc["ds"] > df["ds"].iloc[-1]]["yhat"].tolist()
        return pad_to(raw, n)

    elif name == "ARIMA":
        try:
            fit = ARIMA(df["y"], order=(5, 1, 0)).fit()
        except Exception:
            fit = ARIMA(df["y"], order=(1, 1, 1)).fit()
        return pad_to(list(fit.forecast(steps=n)), n)

    elif name == "SARIMA":
        return sarima_forecast(df, n)

    elif name == "Holt-Winters":
        fit = hw_fit(df["y"])
        return pad_to(list(fit.forecast(steps=n)), n)

    elif name == "LSTM":
        return lstm_forecast(df["y"].values, n, bidir=False, epochs=60)

    elif name == "Bidirectional LSTM":
        return lstm_forecast(df["y"].values, n, bidir=True, epochs=60)

    elif name == "Adaptive LSTM":
        window = min(504, len(df))
        seq    = max(20, min(60, window // 5))
        return lstm_forecast(df["y"].values[-window:], n,
                             bidir=False, seq=seq, epochs=60)

    elif name == "Ensemble":
        return _equal_ensemble_forecast(df, n)

    elif name == "Weighted Ensemble":
        return weighted_ensemble_forecast(df, n)

    raise ValueError(f"unknown model: {name}")


#  Routes

@app.get("/")
def root():
    return {
        "status": "ok",
        "tensorflow_available": TF_AVAILABLE,
        "garch_available": GARCH_AVAILABLE,
        "pmdarima_available": PMDARIMA_AVAILABLE,
    }


@app.post("/stock/history")
def stock_history(req_body: StockReq):
    raw = fetch_raw(req_body.ticker, req_body.start_date, req_body.end_date)
    if raw is None:
        raise HTTPException(404, "No data found for this ticker/date range")
    return {"data": raw.to_dict(orient="records")}


@app.post("/stock/analyze")
def stock_analyze(req_body: StockReq):
    df = fetch_df(req_body.ticker, req_body.start_date, req_body.end_date)
    if df is None or len(df) < 20:
        return {"is_stationary": False, "is_nonlinear": False, "data_points": 0}
    is_stat, is_nl = False, False
    try:
        is_stat = adfuller(df["y"].values)[1] < 0.05
    except Exception:
        pass
    try:
        is_nl = abs(acf(df["y"], nlags=5)[1]) < 0.3
    except Exception:
        pass
    return {"is_stationary": bool(is_stat), "is_nonlinear": bool(is_nl),
            "data_points": len(df)}


@app.post("/stock/indicators")
def stock_indicators(req_body: StockReq):
    df = fetch_df(req_body.ticker, req_body.start_date, req_body.end_date)
    if df is None or len(df) < 30:
        raise HTTPException(404, "Need at least 30 days of data")
    feat = make_features(df)

    def sl(s):
        return [None if (v is None or (isinstance(v, float) and np.isnan(v)))
                else round(float(v), 4) for v in s]

    return {
        "dates":       feat["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "close":       sl(feat["y"]),
        "sma_20":      sl(feat["SMA20"]),
        "sma_50":      sl(feat["SMA50"]),
        "ema_12":      sl(feat["EMA12"]),
        "ema_26":      sl(feat["EMA26"]),
        "rsi":         sl(feat["RSI"]),
        "macd":        sl(feat["MACD"]),
        "macd_signal": sl(feat["MACD_sig"]),
        "macd_hist":   sl(feat["MACD_hist"]),
        "bb_upper":    sl(feat["BB_upper"]),
        "bb_lower":    sl(feat["BB_lower"]),
        "bb_mid":      sl(feat["SMA20"]),
        "volatility":  sl(feat["Vol20"]),
    }

# NEW: /stock/regime/{ticker}
@app.get("/stock/regime/{ticker}")
def stock_regime(ticker: str):
    """
    Full regime analysis for a ticker.
    Uses the last 3 years of daily data.
    Returns volatility regime, Hurst exponent, seasonality,
    ARCH effect, trend, and model recommendation with reasoning.
    """
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=365 * 3)
    df    = fetch_df(ticker, start, end)
    if df is None or len(df) < 50:
        raise HTTPException(404, "Insufficient data for regime analysis (need 50+ trading days)")
    return detect_regime(df)


@app.post("/forecast/evaluate")
def forecast_evaluate(req_body: EvalReq):
    df = fetch_df(req_body.ticker, req_body.start_date, req_body.end_date)
    if df is None:
        raise HTTPException(404, "No data found")
    if len(df) < 100:
        raise HTTPException(400, "Need at least 100 data points (use 2+ year range)")

    split = int(len(df) * 0.80)
    names = ["XGBoost", "Random Forest", "ARIMA", "SARIMA", "Prophet",
             "Holt-Winters", "Ensemble", "Weighted Ensemble"]
    if TF_AVAILABLE:
        names += ["LSTM", "Bidirectional LSTM", "Adaptive LSTM"]

    results = []
    for name in names:
        try:
            yt, yp, dts = run_model_eval(df, name, split)
            k = min(len(yt), len(yp), len(dts))
            if k < 2:
                continue
            yt, yp, dts = yt[:k], yp[:k], dts[:k]
            yta = np.array(yt, float)
            ypa = np.array(yp, float)
            results.append({
                "model":    name,
                "rmse":     round(float(np.sqrt(mean_squared_error(yta, ypa))), 4),
                "mape":     round(mape(yta, ypa), 4),
                "r2":       round(float(r2_score(yta, ypa)), 4),
                "dir_acc":  round(da(yta, ypa), 2),
                "y_true":   [round(float(v), 4) for v in yt],
                "y_pred":   [round(float(v), 4) for v in yp],
                "dates":    dts,
            })
        except Exception as ex:
            print(f"[evaluate/{name}] {ex}")
            traceback.print_exc()
            continue

    if not results:
        raise HTTPException(500, "All models failed. Try a wider date range (2+ years).")
    return results
# UPDATED: /forecast — now returns confidence intervals + regime info
@app.post("/forecast")
def forecast(req_body: ForecastReq):
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=365 * 5)
    df    = fetch_df(req_body.ticker, start, end)
    if df is None or len(df) < 30:
        raise HTTPException(404, "Not enough data to forecast")

    n = req_body.days

    # ── Regime analysis (fast, runs on every forecast call) ───────────────────
    regime_info = {}
    try:
        regime_info = detect_regime(df)
    except Exception as e:
        print(f"[forecast/regime] {e}")

    # ── Auto-model selection ───────────────────────────────────────────────────
    model_name = req_body.model
    if req_body.auto_model and regime_info.get("recommended_model"):
        model_name = regime_info["recommended_model"]
        print(f"[forecast] auto_model → selected {model_name}")

    # ── Run the forecast ───────────────────────────────────────────────────────
    future_dates = pd.date_range(
        start=df["ds"].iloc[-1] + pd.Timedelta(days=1),
        periods=n, freq="B",
    )

    try:
        vals = run_forecast(df, model_name, n)
    except HTTPException:
        raise
    except Exception as ex:
        traceback.print_exc()
        raise HTTPException(500, f"Forecast failed: {ex}")

    vals = pad_to(vals, n)

    # ── GARCH Confidence Intervals ────────────────────────────────────────────
    ci_upper, ci_lower = [], []
    if req_body.use_garch:
        try:
            returns = pd.Series(df["y"].values).pct_change().dropna().values
            ci_upper, ci_lower = garch_confidence_bands(returns, vals, n, confidence=0.90)
        except Exception as e:
            print(f"[forecast/GARCH-CI] {e}")

    return {
        "dates":              [d.strftime("%Y-%m-%d") for d in future_dates],
        "values":             vals,
        "model":              model_name,
        "confidence_upper":   ci_upper,
        "confidence_lower":   ci_lower,
        "regime":             regime_info,
        "garch_available":    GARCH_AVAILABLE,
    }


#Unchanged routes

@app.get("/stock/growth-score/{ticker}")
def growth_score(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        pe   = info.get("trailingPE") or 50
        roe  = info.get("returnOnEquity") or 0
        grow = info.get("earningsQuarterlyGrowth") or 0
        debt = info.get("debtToEquity") or 100
        marg = info.get("profitMargins") or 0
        s = -pe * 0.25 + roe * 100 * 0.3 + grow * 0.25 - debt * 0.1 + marg * 100 * 0.1
        return {"ticker": ticker, "growth_score": round(s, 2)}
    except Exception:
        return {"ticker": ticker, "growth_score": None}


@app.get("/scraper/stocks-under-5")
def cheap_stocks(category: str = "general"):
    urls = {
        "oil":     "https://stock-screener.org/oil-stocks-under-5.aspx",
        "energy":  "https://stock-screener.org/energy-penny-stocks.aspx",
        "general": "https://stock-screener.org/stocks-under-5.aspx",
    }
    try:
        r = req.get(urls.get(category, urls["general"]),
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        out  = []
        for row in soup.select("table tr")[1:]:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) >= 6:
                out.append({"Ticker": cols[0], "Company": cols[1], "Price": cols[5]})
        return out
    except Exception as ex:
        return {"error": str(ex)}


@app.get("/scraper/crypto")
def crypto():
    try:
        r = req.get("https://api.coingecko.com/api/v3/coins/markets",
                    params={"vs_currency": "usd", "order": "market_cap_rank",
                            "per_page": 20, "page": 1}, timeout=15)
        return r.json()
    except Exception as ex:
        return {"error": str(ex)}


@app.get("/news/{ticker}")
def news(ticker: str):
    try:
        q    = ticker.replace(" ", "+")
        feed = feedparser.parse(
            f"https://news.google.com/rss/search?q={q}+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
        )
        pipe = transformers.pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        out = []
        for e in feed.entries[:5]:
            s = pipe(e.title)[0]
            out.append({"title": e.title, "link": e.link,
                        "published": e.published,
                        "sentiment": s["label"], "score": s["score"]})
        return out
    except Exception as ex:
        print(f"[news] {ex}")
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("back_dated4:app", host="0.0.0.0", port=8000, reload=True)
