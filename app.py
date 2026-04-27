"""
================================================================================
  app.py  —  Indian Banking Sector LSTM Forecast Dashboard
  Professional Trader UI built with Streamlit

  Run:
      streamlit run app.py

  First time setup:
      python train_all_banks.py        ← trains & saves all 8 bank models
      streamlit run app.py             ← launch dashboard
================================================================================
"""

import os, json, pickle, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "IndiaBank LSTM Forecaster",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BANKS = {
    "HDFCBANK.NS":   "HDFC Bank",
    "ICICIBANK.NS":  "ICICI Bank",
    "SBIN.NS":       "State Bank of India",
    "KOTAKBANK.NS":  "Kotak Mahindra Bank",
    "AXISBANK.NS":   "Axis Bank",
    "INDUSINDBK.NS": "IndusInd Bank",
    "BANKBARODA.NS": "Bank of Baroda",
    "PNB.NS":        "Punjab National Bank",
}
NIFTY_BANK  = "^NSEBANK"
START_DATE  = "2005-01-01"
MODELS_DIR  = "models"
SEQ_LEN     = 60
FORECAST_DAYS = 7
HIDDEN      = 128
N_LAYERS    = 2
DROPOUT     = 0.2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "ret", "ma_10", "ma_20", "ma_50",
    "dist_ma20", "dist_ma50", "dist_ma200",
    "vol_10", "vol_20",
    "rsi", "macd", "macd_sig", "macd_hist",
    "bb_pos", "bb_width", "vol_ratio",
    "nifty_ret", "nifty_dist",
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
    "Close",
]
TARGET_IDX = FEATURE_COLS.index("Close")
N_FEATURES = len(FEATURE_COLS)

# ─────────────────────────────────────────────────────────────────────────────
# DARK THEME COLORS
# ─────────────────────────────────────────────────────────────────────────────
BG    = "#0d0d0d"
AX    = "#141414"
BD    = "#2a2a2a"
BLUE  = "#00D4FF"
GREEN = "#00FF88"
RED   = "#FF4466"
AMBER = "#FFB300"
GRAY  = "#666666"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark professional look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0a0a0a; color: #e0e0e0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * { color: #cccccc !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #141414;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"]  { color: #888 !important; font-size: 0.75rem; }
[data-testid="stMetricValue"]  { color: #00D4FF !important; font-size: 1.4rem; font-weight: 700; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF22, #00D4FF11);
    color: #00D4FF;
    border: 1px solid #00D4FF55;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00D4FF44, #00D4FF22);
    border-color: #00D4FF;
    transform: translateY(-1px);
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #141414 !important;
    border: 1px solid #333 !important;
    color: #e0e0e0 !important;
    border-radius: 8px;
}

/* Dividers */
hr { border-color: #222 !important; }

/* Headers */
h1, h2, h3 { color: #e0e0e0 !important; }

/* Success / info / error boxes */
.stSuccess { background-color: #00FF8811 !important; border-color: #00FF88 !important; }
.stInfo    { background-color: #00D4FF11 !important; border-color: #00D4FF !important; }
.stError   { background-color: #FF446611 !important; border-color: #FF4466 !important; }
.stWarning { background-color: #FFB30011 !important; border-color: #FFB300 !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #222; border-radius: 8px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #111; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #888; }
.stTabs [aria-selected="true"] { color: #00D4FF !important; border-bottom-color: #00D4FF !important; }

/* Progress bar */
.stProgress > div > div { background-color: #00D4FF !important; }

/* Spinner */
.stSpinner > div { border-top-color: #00D4FF !important; }

/* Chart figures */
.element-container iframe, .element-container img { border-radius: 10px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION  (must match train_all_banks.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(N_FEATURES, HIDDEN, N_LAYERS,
                            dropout=DROPOUT if N_LAYERS > 1 else 0,
                            batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN)
        self.drop = nn.Dropout(DROPOUT)
        self.fc   = nn.Sequential(
            nn.Linear(HIDDEN, 64), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.norm(out[:, -1]))).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def model_exists(ticker):
    d = os.path.join(MODELS_DIR, ticker)
    return all(os.path.exists(os.path.join(d, f))
               for f in ["model.pt", "scaler.pkl", "price_scaler.pkl", "meta.json"])


@st.cache_resource(show_spinner=False)
def load_model(ticker):
    d = os.path.join(MODELS_DIR, ticker)
    model = StockLSTM().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(d, "model.pt"),
                                     map_location=DEVICE))
    model.eval()
    with open(os.path.join(d, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(d, "price_scaler.pkl"), "rb") as f:
        price_scaler = pickle.load(f)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    return model, scaler, price_scaler, meta


@st.cache_data(show_spinner=False, ttl=300)
def fetch_data(ticker, start=START_DATE):
    end = datetime.today().strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw.dropna(inplace=True)
    return raw


def add_features(df, nifty):
    f  = df.copy()
    c  = f["Close"]
    f["ret"]        = c.pct_change()
    f["ma_10"]      = c.rolling(10).mean()
    f["ma_20"]      = c.rolling(20).mean()
    f["ma_50"]      = c.rolling(50).mean()
    f["ma_200"]     = c.rolling(200).mean()
    f["dist_ma20"]  = (c - f["ma_20"])  / f["ma_20"]
    f["dist_ma50"]  = (c - f["ma_50"])  / f["ma_50"]
    f["dist_ma200"] = (c - f["ma_200"]) / f["ma_200"]
    f["vol_10"]     = f["ret"].rolling(10).std()
    f["vol_20"]     = f["ret"].rolling(20).std()
    delta           = c.diff()
    gain            = delta.clip(lower=0).rolling(14).mean()
    loss            = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"]        = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12           = c.ewm(span=12, adjust=False).mean()
    ema26           = c.ewm(span=26, adjust=False).mean()
    f["macd"]       = ema12 - ema26
    f["macd_sig"]   = f["macd"].ewm(span=9, adjust=False).mean()
    f["macd_hist"]  = f["macd"] - f["macd_sig"]
    bm              = c.rolling(20).mean()
    bs              = c.rolling(20).std()
    f["bb_upper"]   = bm + 2 * bs
    f["bb_lower"]   = bm - 2 * bs
    f["bb_pos"]     = (c - f["bb_lower"]) / (f["bb_upper"] - f["bb_lower"] + 1e-9)
    f["bb_width"]   = (f["bb_upper"] - f["bb_lower"]) / bm
    f["vol_ratio"]  = f["Volume"] / f["Volume"].rolling(20).mean()
    f["nifty_ret"]  = nifty["Close"].pct_change()
    nifty_ma20      = nifty["Close"].rolling(20).mean()
    f["nifty_dist"] = (nifty["Close"] - nifty_ma20) / nifty_ma20
    for lag in [1, 2, 3, 5]:
        f[f"ret_lag{lag}"] = f["ret"].shift(lag)
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    f.dropna(inplace=True)
    return f


def inverse_price(arr, price_scaler):
    return price_scaler.inverse_transform(
        np.array(arr, dtype=np.float32).reshape(-1, 1)).flatten()


def get_forecast_dates(last_date, n=7):
    dates, d = [], last_date
    while len(dates) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d)
    return dates


def style_fig(fig):
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(AX)
        ax.tick_params(colors=GRAY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(BD)
        ax.yaxis.label.set_color(GRAY)
        ax.xaxis.label.set_color(GRAY)
        ax.title.set_color("white")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(ticker, model, scaler, price_scaler, meta):
    bank_df  = fetch_data(ticker)
    nifty_df = fetch_data(NIFTY_BANK)
    common   = bank_df.index.intersection(nifty_df.index)
    bank_df  = bank_df.loc[common]
    nifty_df = nifty_df.loc[common]
    df       = add_features(bank_df, nifty_df)

    data_arr = df[FEATURE_COLS].values
    data_sc  = scaler.transform(data_arr)

    # Test set actual vs predicted
    split      = int(len(data_sc) * 0.85)
    X_test_all = np.array([data_sc[i: i+SEQ_LEN]
                            for i in range(split, len(data_sc)-SEQ_LEN)],
                           dtype=np.float32)
    prices_test = df["Close"].values[SEQ_LEN+split: SEQ_LEN+split+len(X_test_all)]
    dates_test  = df.index[SEQ_LEN+split: SEQ_LEN+split+len(X_test_all)]

    model.eval()
    with torch.no_grad():
        x_t       = torch.tensor(X_test_all).to(DEVICE)
        preds_sc  = model(x_t).cpu().numpy()
    preds_inr = inverse_price(preds_sc, price_scaler)

    # 7-day forecast
    window = data_sc[-SEQ_LEN:].copy()
    fc_prices = []
    for _ in range(FORECAST_DAYS):
        x_t = torch.tensor(window[np.newaxis], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            p = float(model(x_t).cpu().numpy()[0])
        fc_prices.append(p)
        new_row             = window[-1].copy()
        new_row[TARGET_IDX] = p
        window              = np.vstack([window[1:], new_row])

    fc_prices = inverse_price(fc_prices, price_scaler)
    fc_dates  = get_forecast_dates(df.index[-1])

    # Metrics
    mae  = mean_absolute_error(prices_test, preds_inr)
    rmse = np.sqrt(mean_squared_error(prices_test, preds_inr))
    mape = np.mean(np.abs((prices_test - preds_inr) / prices_test)) * 100
    r2   = r2_score(prices_test, preds_inr)

    return {
        "df": df,
        "dates_test":  dates_test,
        "prices_test": prices_test,
        "preds_inr":   preds_inr,
        "fc_dates":    fc_dates,
        "fc_prices":   fc_prices,
        "mae": mae, "rmse": rmse, "mape": mape, "r2": r2,
        "acc": max(0, 100 - mape),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def chart_actual_vs_predicted(res, ticker):
    dt   = res["dates_test"]
    act  = res["prices_test"]
    pred = res["preds_inr"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG)

    # Panel 1: actual vs predicted
    ax1 = axes[0]; ax1.set_facecolor(AX)
    ax1.plot(dt, act,  color=BLUE,  lw=1.8, label="Actual Price")
    ax1.plot(dt, pred, color=AMBER, lw=1.4, ls="--", alpha=0.9, label="Predicted Price")
    ax1.fill_between(dt, act, pred,
                     where=pred >= act, alpha=0.10, color=GREEN, label="Over-predicted")
    ax1.fill_between(dt, act, pred,
                     where=pred <  act, alpha=0.10, color=RED,   label="Under-predicted")
    ax1.set_ylabel("Price (₹)", color=GRAY, fontsize=9)
    info = (f"MAE ₹{res['mae']:.0f}  |  RMSE ₹{res['rmse']:.0f}  |  "
            f"MAPE {res['mape']:.2f}%  |  R² {res['r2']:.4f}  |  Accuracy ≈ {res['acc']:.1f}%")
    ax1.text(0.01, 0.97, info, transform=ax1.transAxes,
             color=AMBER, fontsize=8, va="top",
             bbox=dict(boxstyle="round", facecolor="#111", alpha=0.7))
    ax1.set_title("Actual vs Predicted Close Price (Test Set)",
                  color="white", fontsize=11, fontweight="bold")
    ax1.legend(facecolor=AX, labelcolor="white", fontsize=8, loc="upper left")
    ax1.tick_params(colors=GRAY, labelsize=8)
    for sp in ax1.spines.values(): sp.set_color(BD)

    # Panel 2: last 90 days zoom
    ax2 = axes[1]; ax2.set_facecolor(AX)
    n = min(90, len(dt))
    ax2.plot(dt[-n:], act[-n:],  color=BLUE,  lw=2,   label="Actual")
    ax2.plot(dt[-n:], pred[-n:], color=AMBER, lw=1.8, ls="--", label="Predicted")
    ax2.fill_between(dt[-n:], act[-n:], pred[-n:],
                     where=pred[-n:] >= act[-n:], alpha=0.12, color=GREEN)
    ax2.fill_between(dt[-n:], act[-n:], pred[-n:],
                     where=pred[-n:] <  act[-n:], alpha=0.12, color=RED)
    ax2.set_ylabel("Price (₹)", color=GRAY, fontsize=9)
    ax2.set_title("Zoom — Last 90 Trading Days",
                  color="white", fontsize=11, fontweight="bold")
    ax2.legend(facecolor=AX, labelcolor="white", fontsize=8)
    ax2.tick_params(colors=GRAY, labelsize=8)
    for sp in ax2.spines.values(): sp.set_color(BD)

    plt.tight_layout()
    return fig


def chart_forecast(res, ticker, bank_name):
    hist_n      = 90
    hist_dates  = res["df"].index[-hist_n:]
    hist_prices = res["df"]["Close"].values[-hist_n:]
    last_price  = hist_prices[-1]
    fc_dates    = res["fc_dates"]
    fc_prices   = res["fc_prices"]

    bridge_d = [hist_dates[-1]] + fc_dates
    bridge_p = [last_price]     + list(fc_prices)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

    # Left: line chart
    ax1 = axes[0]; ax1.set_facecolor(AX)
    ax1.plot(hist_dates, hist_prices,
             color=BLUE, lw=2, label=f"Actual (last {hist_n} days)")
    ax1.plot(bridge_d, bridge_p,
             color=AMBER, lw=2.5, ls="--", marker="o",
             markersize=7, markerfacecolor=AMBER, label="7-Day Forecast", zorder=5)
    if fc_dates:
        ax1.axvspan(fc_dates[0], fc_dates[-1], alpha=0.06, color=AMBER)
    ax1.axvline(hist_dates[-1], color="white", lw=1, ls=":", alpha=0.4)
    ax1.annotate(f"₹{last_price:,.0f}", xy=(hist_dates[-1], last_price),
                 xytext=(-70, 15), textcoords="offset points", color="white",
                 fontsize=8, arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
    ax1.annotate(f"₹{fc_prices[-1]:,.0f}", xy=(fc_dates[-1], fc_prices[-1]),
                 xytext=(8, -20), textcoords="offset points", color=AMBER,
                 fontsize=9, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.8))
    ax1.set_ylabel("Price (₹)", color=GRAY, fontsize=9)
    ax1.set_title(f"{bank_name} — 7-Day Price Forecast",
                  color="white", fontsize=11, fontweight="bold")
    ax1.legend(facecolor=AX, labelcolor="white", fontsize=8)
    ax1.tick_params(colors=GRAY, labelsize=8)
    for sp in ax1.spines.values(): sp.set_color(BD)

    # Right: bar chart by day
    ax2 = axes[1]; ax2.set_facecolor(AX)
    day_labels = [f"D{i+1}\n{d.strftime('%d %b')}" for i, d in enumerate(fc_dates)]
    bar_colors = [GREEN if p >= last_price else RED for p in fc_prices]
    bars       = ax2.bar(day_labels, fc_prices, color=bar_colors,
                         alpha=0.85, edgecolor="none", width=0.55)
    ax2.axhline(last_price, color="white", lw=1.5, ls="--",
                alpha=0.6, label=f"Current ₹{last_price:,.0f}")
    ymin = min(fc_prices) * 0.998
    ymax = max(fc_prices) * 1.002
    ax2.set_ylim(ymin, ymax)
    for bar, price in zip(bars, fc_prices):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (ymax - ymin) * 0.003,
                 f"₹{price:,.0f}", ha="center",
                 color="white", fontsize=7.5, fontweight="bold")
    ax2.set_title("Day-by-Day Forecast",
                  color="white", fontsize=11, fontweight="bold")
    ax2.legend(facecolor=AX, labelcolor="white", fontsize=8)
    ax2.tick_params(colors=GRAY, labelsize=8)
    for sp in ax2.spines.values(): sp.set_color(BD)

    plt.tight_layout()
    return fig


def chart_technicals(df, ticker):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=BG,
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    n   = min(504, len(df))
    sub = df.iloc[-n:]

    # Price + BBs + MAs
    ax1 = axes[0]; ax1.set_facecolor(AX)
    ax1.plot(sub.index, sub["Close"],    color=BLUE,  lw=1.8, label="Close")
    ax1.plot(sub.index, sub["ma_20"],    color=AMBER, lw=1,   ls="--", alpha=0.7, label="MA-20")
    ax1.plot(sub.index, sub["ma_50"],    color=GREEN, lw=1,   ls="--", alpha=0.7, label="MA-50")
    ax1.plot(sub.index, sub["bb_upper"], color=GRAY,  lw=0.8, ls=":",  alpha=0.6, label="BB Upper")
    ax1.plot(sub.index, sub["bb_lower"], color=GRAY,  lw=0.8, ls=":",  alpha=0.6, label="BB Lower")
    ax1.fill_between(sub.index, sub["bb_upper"], sub["bb_lower"],
                     alpha=0.05, color=BLUE)
    ax1.set_ylabel("Price (₹)", color=GRAY, fontsize=9)
    ax1.set_title("Price + Bollinger Bands + Moving Averages",
                  color="white", fontsize=10, fontweight="bold")
    ax1.legend(facecolor=AX, labelcolor="white", fontsize=7, ncol=3)
    ax1.tick_params(colors=GRAY, labelsize=8)
    for sp in ax1.spines.values(): sp.set_color(BD)

    # RSI
    ax2 = axes[1]; ax2.set_facecolor(AX)
    ax2.plot(sub.index, sub["rsi"], color=BLUE, lw=1.2)
    ax2.axhline(70, color=RED,   lw=1, ls="--", alpha=0.7, label="Overbought 70")
    ax2.axhline(30, color=GREEN, lw=1, ls="--", alpha=0.7, label="Oversold 30")
    ax2.axhline(50, color=GRAY,  lw=0.6, ls=":", alpha=0.4)
    ax2.fill_between(sub.index, sub["rsi"], 70,
                     where=sub["rsi"] >= 70, alpha=0.15, color=RED)
    ax2.fill_between(sub.index, sub["rsi"], 30,
                     where=sub["rsi"] <= 30, alpha=0.15, color=GREEN)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color=GRAY, fontsize=9)
    ax2.legend(facecolor=AX, labelcolor="white", fontsize=7)
    ax2.tick_params(colors=GRAY, labelsize=8)
    for sp in ax2.spines.values(): sp.set_color(BD)

    # MACD
    ax3 = axes[2]; ax3.set_facecolor(AX)
    ax3.plot(sub.index, sub["macd"],     color=BLUE,  lw=1.2, label="MACD")
    ax3.plot(sub.index, sub["macd_sig"], color=AMBER, lw=1.2, ls="--", label="Signal")
    bar_c = [GREEN if v >= 0 else RED for v in sub["macd_hist"]]
    ax3.bar(sub.index, sub["macd_hist"], color=bar_c, alpha=0.6, label="Histogram")
    ax3.axhline(0, color=GRAY, lw=0.6, ls=":", alpha=0.5)
    ax3.set_ylabel("MACD", color=GRAY, fontsize=9)
    ax3.legend(facecolor=AX, labelcolor="white", fontsize=7)
    ax3.tick_params(colors=GRAY, labelsize=8)
    for sp in ax3.spines.values(): sp.set_color(BD)

    plt.tight_layout()
    return fig


def chart_error(res):
    act  = res["prices_test"]
    pred = res["preds_inr"]
    dt   = res["dates_test"]
    err  = act - pred
    pct  = (err / act) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor=BG)

    # Error over time
    ax1 = axes[0]; ax1.set_facecolor(AX)
    ax1.plot(dt, pct, color=RED, lw=0.8, alpha=0.8)
    ax1.axhline(0,         color="white", lw=0.8, ls="--", alpha=0.4)
    ax1.axhline(pct.mean(),color=AMBER,  lw=1,   ls="--",
                label=f"Mean {pct.mean():.2f}%")
    ax1.fill_between(dt, pct, 0, where=pct > 0, color=GREEN, alpha=0.12)
    ax1.fill_between(dt, pct, 0, where=pct < 0, color=RED,   alpha=0.12)
    ax1.set_ylabel("Error (%)", color=GRAY, fontsize=9)
    ax1.set_title("Prediction Error % Over Time",
                  color="white", fontsize=10, fontweight="bold")
    ax1.legend(facecolor=AX, labelcolor="white", fontsize=8)
    ax1.tick_params(colors=GRAY, labelsize=8)
    for sp in ax1.spines.values(): sp.set_color(BD)

    # Distribution
    ax2 = axes[1]; ax2.set_facecolor(AX)
    ax2.hist(pct, bins=60, color=BLUE, alpha=0.75, edgecolor="none", density=True)
    ax2.axvline(0,          color="white", lw=1,   ls="--", alpha=0.5)
    ax2.axvline(pct.mean(), color=AMBER,  lw=1.5, ls="--",
                label=f"Mean {pct.mean():.2f}%")
    ax2.set_xlabel("Prediction Error (%)", color=GRAY, fontsize=9)
    ax2.set_ylabel("Density",              color=GRAY, fontsize=9)
    ax2.set_title("Distribution of Prediction Error",
                  color="white", fontsize=10, fontweight="bold")
    ax2.legend(facecolor=AX, labelcolor="white", fontsize=8)
    ax2.tick_params(colors=GRAY, labelsize=8)
    for sp in ax2.spines.values(): sp.set_color(BD)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE MODEL (retrain with latest data)
# ─────────────────────────────────────────────────────────────────────────────
def retrain_bank(ticker, bank_name, progress_bar, status_text):
    from torch.utils.data import Dataset, DataLoader

    class SeqDS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X)
            self.y = torch.tensor(y)
        def __len__(self):         return len(self.X)
        def __getitem__(self, i):  return self.X[i], self.y[i]

    status_text.text("📥 Downloading latest data...")
    bank_df  = fetch_data(ticker)
    nifty_df = fetch_data(NIFTY_BANK)
    common   = bank_df.index.intersection(nifty_df.index)
    df       = add_features(bank_df.loc[common], nifty_df.loc[common])

    status_text.text("⚙️ Preparing sequences...")
    data_arr     = df[FEATURE_COLS].values
    scaler       = MinMaxScaler()
    data_sc      = scaler.fit_transform(data_arr)
    price_scaler = MinMaxScaler()
    price_scaler.fit(df[["Close"]].values)

    X_all, y_all = [], []
    for i in range(len(data_sc) - SEQ_LEN):
        X_all.append(data_sc[i: i+SEQ_LEN])
        y_all.append(data_sc[i+SEQ_LEN, TARGET_IDX])
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)

    split    = int(len(X_all) * 0.85)
    X_tr, X_te = X_all[:split], X_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]

    tr_dl = DataLoader(SeqDS(X_tr, y_tr), batch_size=64, shuffle=True)
    te_dl = DataLoader(SeqDS(X_te, y_te), batch_size=64)

    EPOCHS  = 40
    model   = StockLSTM().to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit    = nn.MSELoss()
    best_vl = float("inf")
    best_st = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in te_dl:
                vl += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        avg_vl = vl / len(te_dl)
        sch.step(avg_vl)
        if avg_vl < best_vl:
            best_vl = avg_vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        progress_bar.progress(ep / EPOCHS)
        status_text.text(f"🏋️ Training epoch {ep}/{EPOCHS}  val_loss={avg_vl:.5f}")

    model.load_state_dict(best_st)

    # Metrics
    model.eval()
    prices_test = df["Close"].values[SEQ_LEN:][split:]
    preds_sc = []
    with torch.no_grad():
        x_t = torch.tensor(X_te).to(DEVICE)
        preds_sc = model(x_t).cpu().numpy()
    preds_inr = price_scaler.inverse_transform(
        preds_sc.reshape(-1, 1)).flatten()
    mae  = mean_absolute_error(prices_test, preds_inr)
    rmse = np.sqrt(mean_squared_error(prices_test, preds_inr))
    mape = np.mean(np.abs((prices_test - preds_inr) / prices_test)) * 100
    r2   = r2_score(prices_test, preds_inr)

    # Save
    save_dir = os.path.join(MODELS_DIR, ticker)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, "price_scaler.pkl"), "wb") as f:
        pickle.dump(price_scaler, f)

    END_DATE = datetime.today().strftime("%Y-%m-%d")
    meta = {
        "ticker": ticker, "name": bank_name,
        "trained_on": END_DATE,
        "data_start": str(df.index[0].date()),
        "data_end":   str(df.index[-1].date()),
        "mae": round(mae, 2), "rmse": round(rmse, 2),
        "mape": round(mape, 2), "r2": round(r2, 4),
        "accuracy": round(max(0, 100 - mape), 2),
        "last_close": round(float(df["Close"].iloc[-1]), 2),
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN, "n_features": N_FEATURES,
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Clear cached model so it reloads
    load_model.clear()
    fetch_data.clear()

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2rem;'>📈</div>
        <div style='font-size:1.1rem; font-weight:700; color:#00D4FF;'>IndiaBank LSTM</div>
        <div style='font-size:0.7rem; color:#555; margin-top:4px;'>AI Stock Forecaster</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏦 Select Bank")
    bank_options = list(BANKS.items())   # [(ticker, name), ...]
    names        = [n for _, n in bank_options]
    tickers      = [t for t, _ in bank_options]

    selected_name   = st.selectbox("", names, label_visibility="collapsed")
    selected_ticker = tickers[names.index(selected_name)]

    # Model status
    exists = model_exists(selected_ticker)
    if exists:
        with open(os.path.join(MODELS_DIR, selected_ticker, "meta.json")) as f:
            cached_meta = json.load(f)
        st.markdown(f"""
        <div style='background:#0a1a0a; border:1px solid #1a3a1a;
                    border-radius:8px; padding:10px; margin:10px 0;'>
            <div style='color:#00FF88; font-size:0.75rem; font-weight:700;'>✅ MODEL READY</div>
            <div style='color:#555; font-size:0.68rem; margin-top:4px;'>
                Trained: {cached_meta.get('trained_on','—')}<br>
                Data: {cached_meta.get('data_start','—')} → {cached_meta.get('data_end','—')}<br>
                Accuracy: {cached_meta.get('accuracy','—')}%  |  R²: {cached_meta.get('r2','—')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No saved model found.\nRun `train_all_banks.py` first.")

    st.markdown("---")

    # PREDICT button
    predict_btn = st.button("🔮  Run Forecast", use_container_width=True,
                            disabled=not exists)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # UPDATE button
    update_btn = st.button("🔄  Update Model (sync new data)",
                           use_container_width=True)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.7rem; color:#444; line-height:1.8;'>
        <b style='color:#555'>Model config</b><br>
        Architecture : LSTM × 2 layers<br>
        Hidden units : 128<br>
        Sequence len : {SEQ_LEN} days<br>
        Features     : {N_FEATURES}<br>
        Device       : {DEVICE}<br>
        Forecast     : {FORECAST_DAYS} days
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.65rem; color:#333; text-align:center;'>
        BTech Capstone Project<br>
        Indian Banking Sector LSTM
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────
# Header
st.markdown(f"""
<div style='display:flex; align-items:center; justify-content:space-between;
            border-bottom:1px solid #1a1a1a; padding-bottom:16px; margin-bottom:20px;'>
    <div>
        <h1 style='margin:0; font-size:1.6rem; color:#e0e0e0;'>
            📈 {selected_name}
            <span style='color:#555; font-size:1rem; font-weight:400;'>
              ({selected_ticker})
            </span>
        </h1>
        <div style='color:#555; font-size:0.8rem; margin-top:4px;'>
            Indian Banking Sector · LSTM Stock Price Forecaster · 20-Year Data
        </div>
    </div>
    <div style='text-align:right; color:#444; font-size:0.75rem;'>
        {datetime.now().strftime('%a, %d %b %Y  %H:%M')}
    </div>
</div>
""", unsafe_allow_html=True)


# ── UPDATE MODEL flow ─────────────────────────────────────────────────────────
if update_btn:
    st.markdown("### 🔄 Updating Model with Latest Data")
    prog  = st.progress(0)
    stat  = st.empty()
    try:
        new_meta = retrain_bank(selected_ticker, selected_name, prog, stat)
        prog.progress(1.0)
        stat.empty()
        st.success(
            f"✅ Model updated successfully!  "
            f"New accuracy: **{new_meta['accuracy']:.1f}%**  |  "
            f"Data through: **{new_meta['data_end']}**"
        )
    except Exception as e:
        stat.empty()
        st.error(f"❌ Update failed: {e}")
    st.rerun()


# ── PREDICT flow ──────────────────────────────────────────────────────────────
if predict_btn and exists:
    with st.spinner("Loading model and running prediction..."):
        model, scaler, price_scaler, meta = load_model(selected_ticker)
        res = run_prediction(selected_ticker, model, scaler, price_scaler, meta)

    # ── Live price ticker row ────────────────────────────────────────────
    last_close = float(res["df"]["Close"].iloc[-1])
    prev_close = float(res["df"]["Close"].iloc[-2])
    day_chg    = last_close - prev_close
    day_pct    = day_chg / prev_close * 100
    week_ret   = (last_close / float(res["df"]["Close"].iloc[-6]) - 1) * 100
    month_ret  = (last_close / float(res["df"]["Close"].iloc[-22]) - 1) * 100
    year_ret   = (last_close / float(res["df"]["Close"].iloc[-252]) - 1) * 100
    fc7_chg    = res["fc_prices"][-1] - last_close
    fc7_pct    = fc7_chg / last_close * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Current Price",   f"₹{last_close:,.2f}",
              f"{day_chg:+.2f} ({day_pct:+.2f}%)")
    c2.metric("7-Day Forecast",  f"₹{res['fc_prices'][-1]:,.2f}",
              f"{fc7_chg:+.2f} ({fc7_pct:+.2f}%)")
    c3.metric("1-Week Return",   f"{week_ret:+.2f}%")
    c4.metric("1-Month Return",  f"{month_ret:+.2f}%")
    c5.metric("1-Year Return",   f"{year_ret:+.2f}%")
    c6.metric("Model Accuracy",  f"{res['acc']:.1f}%",
              f"R² = {res['r2']:.4f}")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── 7-day forecast table ─────────────────────────────────────────────
    st.markdown("### 🔮 7-Day Price Forecast")
    tbl_rows = []
    prev = last_close
    for i, (d, p) in enumerate(zip(res["fc_dates"], res["fc_prices"])):
        chg = p - prev
        pct = chg / prev * 100
        sig = "🟢 BUY" if chg > 0 else "🔴 SELL"
        tbl_rows.append({
            "Day":     f"Day {i+1}",
            "Date":    d.strftime("%a, %d %b %Y"),
            "Forecast (₹)": f"₹{p:,.2f}",
            "Change (₹)":   f"{chg:+.2f}",
            "Change (%)":   f"{pct:+.2f}%",
            "Signal":       sig,
        })
        prev = p
    tbl_df = pd.DataFrame(tbl_rows)
    st.dataframe(tbl_df, use_container_width=True, hide_index=True)

    # ── TABS ────────────────────────────────────────────────────────────
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Actual vs Predicted",
        "🔮 Forecast Chart",
        "📉 Technical Indicators",
        "📋 Error Analysis",
    ])

    with tab1:
        st.markdown(f"""
        <div style='display:flex; gap:20px; margin-bottom:12px;'>
            <div style='background:#141414; border:1px solid #222;
                        border-radius:8px; padding:12px 20px; flex:1; text-align:center;'>
                <div style='color:#888; font-size:0.72rem;'>MAE</div>
                <div style='color:#00D4FF; font-size:1.3rem; font-weight:700;'>
                    ₹{res['mae']:.2f}</div>
            </div>
            <div style='background:#141414; border:1px solid #222;
                        border-radius:8px; padding:12px 20px; flex:1; text-align:center;'>
                <div style='color:#888; font-size:0.72rem;'>RMSE</div>
                <div style='color:#FFB300; font-size:1.3rem; font-weight:700;'>
                    ₹{res['rmse']:.2f}</div>
            </div>
            <div style='background:#141414; border:1px solid #222;
                        border-radius:8px; padding:12px 20px; flex:1; text-align:center;'>
                <div style='color:#888; font-size:0.72rem;'>MAPE</div>
                <div style='color:#FF4466; font-size:1.3rem; font-weight:700;'>
                    {res['mape']:.2f}%</div>
            </div>
            <div style='background:#141414; border:1px solid #222;
                        border-radius:8px; padding:12px 20px; flex:1; text-align:center;'>
                <div style='color:#888; font-size:0.72rem;'>R² Score</div>
                <div style='color:#00FF88; font-size:1.3rem; font-weight:700;'>
                    {res['r2']:.4f}</div>
            </div>
            <div style='background:#141414; border:1px solid #222;
                        border-radius:8px; padding:12px 20px; flex:1; text-align:center;'>
                <div style='color:#888; font-size:0.72rem;'>Accuracy</div>
                <div style='color:#00FF88; font-size:1.3rem; font-weight:700;'>
                    {res['acc']:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(chart_actual_vs_predicted(res, selected_ticker))

    with tab2:
        st.pyplot(chart_forecast(res, selected_ticker, selected_name))

    with tab3:
        st.pyplot(chart_technicals(res["df"], selected_ticker))

    with tab4:
        st.pyplot(chart_error(res))

elif not predict_btn and exists:
    # Landing state — show quick overview
    st.markdown("""
    <div style='display:flex; align-items:center; justify-content:center;
                height:320px; flex-direction:column; gap:16px;
                background:#0d0d0d; border:1px solid #1a1a1a; border-radius:12px;'>
        <div style='font-size:3rem;'>🔮</div>
        <div style='font-size:1.2rem; color:#555; font-weight:500;'>
            Select a bank and click <span style='color:#00D4FF;'>Run Forecast</span>
        </div>
        <div style='font-size:0.8rem; color:#333;'>
            Model will predict the next 7 trading days using LSTM
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show all bank statuses
    st.markdown("### 🏦 Model Status — All Banks")
    cols = st.columns(4)
    for i, (t, n) in enumerate(BANKS.items()):
        with cols[i % 4]:
            ok = model_exists(t)
            if ok:
                with open(os.path.join(MODELS_DIR, t, "meta.json")) as f:
                    m = json.load(f)
                st.markdown(f"""
                <div style='background:#0a1a0a; border:1px solid #1a3a1a;
                            border-radius:8px; padding:10px; margin-bottom:8px;'>
                    <div style='color:#00FF88; font-size:0.72rem; font-weight:700;'>
                        ✅ {n}</div>
                    <div style='color:#555; font-size:0.65rem; margin-top:4px;'>
                        Acc: {m.get('accuracy','—')}%  R²: {m.get('r2','—')}<br>
                        ₹{m.get('last_close','—')}  |  {m.get('data_end','—')}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:#1a0a0a; border:1px solid #3a1a1a;
                            border-radius:8px; padding:10px; margin-bottom:8px;'>
                    <div style='color:#FF4466; font-size:0.72rem; font-weight:700;'>
                        ❌ {n}</div>
                    <div style='color:#555; font-size:0.65rem; margin-top:4px;'>
                        Not trained yet</div>
                </div>""", unsafe_allow_html=True)

elif not exists:
    st.error(
        "⚠️ No trained model found for this bank.  \n"
        "Run `python train_all_banks.py` to train all 8 bank models first."
    )
