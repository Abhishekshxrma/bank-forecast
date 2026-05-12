import os, json, pickle, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
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
    page_title  = "INDIABANK · LSTM FORECASTER",
    page_icon   = "▸",
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
# BLOOMBERG TERMINAL COLOR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#000000"          # Pure black terminal background
PANEL   = "#080808"          # Panel / card background
BORDER  = "#1C1C1C"          # Subtle border
BORDER2 = "#2A2A2A"          # Slightly brighter border
ORANGE  = "#F7921D"          # Bloomberg orange — primary accent
CYAN    = "#00B8FF"          # Data / actual price lines
GREEN   = "#00C853"          # Positive / up
RED     = "#FF3B3B"          # Negative / down
AMBER   = "#FFB800"          # Forecast / predicted
TEXT    = "#D4D4D4"          # Primary text
DIM     = "#555555"          # Dimmed text
LABEL   = "#777777"          # Field labels

# matplotlib aliases
AX      = "#050505"
BD      = "#1E1E1E"
GRAY    = "#666666"
BLUE    = CYAN

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB  — set monospace font globally
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "monospace",
    "axes.facecolor":     AX,
    "figure.facecolor":   BG,
    "axes.edgecolor":     BD,
    "axes.labelcolor":    GRAY,
    "xtick.color":        GRAY,
    "ytick.color":        GRAY,
    "grid.color":         "#111111",
    "grid.linewidth":     0.5,
    "axes.grid":          True,
    "grid.alpha":         0.6,
    "text.color":         TEXT,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ─────────────────────────────────────────────────────────────────────────────
# BLOOMBERG TERMINAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background-color: #000000;
    color: #D4D4D4;
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #050505;
    border-right: 1px solid #1C1C1C;
}
[data-testid="stSidebar"] * {
    color: #AAAAAA !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* ── Streamlit metric cards — override to hidden, we use custom HTML ── */
[data-testid="stMetric"] {
    background-color: #080808;
    border: 1px solid #1C1C1C;
    border-top: 2px solid #F7921D;
    border-radius: 0;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] {
    color: #777777 !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: #F7921D !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] {
    font-size: 0.72rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricDeltaIcon-Up"]   { color: #00C853 !important; }
[data-testid="stMetricDeltaIcon-Down"] { color: #FF3B3B !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent;
    color: #F7921D;
    border: 1px solid #F7921D;
    border-radius: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 8px 16px;
    transition: background 0.15s, color 0.15s;
    width: 100%;
}
.stButton > button:hover {
    background: #F7921D;
    color: #000000;
    transform: none;
    box-shadow: none;
}
.stButton > button:disabled {
    border-color: #2A2A2A;
    color: #333333;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background-color: #080808 !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 0 !important;
    color: #D4D4D4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stSelectbox label {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #555555 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #000;
    border-bottom: 1px solid #1C1C1C;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #555555;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 0;
    padding: 8px 18px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #F7921D !important;
    border-bottom: 2px solid #F7921D !important;
    background: transparent !important;
}

/* ── Dividers ── */
hr { border: none; border-top: 1px solid #1C1C1C !important; margin: 16px 0; }

/* ── Headers ── */
h1, h2, h3 {
    color: #D4D4D4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.02em;
}

/* ── Progress bar ── */
.stProgress > div > div { background-color: #F7921D !important; border-radius: 0 !important; }
.stProgress > div { background-color: #111 !important; border-radius: 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #F7921D !important; }

/* ── Alerts ── */
.stSuccess {
    background-color: #00C85311 !important;
    border: 1px solid #00C853 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}
.stInfo {
    background-color: #00B8FF0D !important;
    border: 1px solid #00B8FF44 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stError {
    background-color: #FF3B3B0D !important;
    border: 1px solid #FF3B3B44 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}
.stWarning {
    background-color: #FFB8000D !important;
    border: 1px solid #FFB80044 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1C1C1C !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #2A2A2A; }
::-webkit-scrollbar-thumb:hover { background: #F7921D; }

/* ── Global text ── */
p, span, div, label {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Chart images ── */
.element-container iframe,
.element-container img { border-radius: 0 !important; }

/* ── Status indicator blink ── */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
.live-dot { animation: blink 1.4s ease-in-out infinite; }

/* ── Scanline overlay effect on header ── */
.bb-header::after {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.06) 2px,
        rgba(0,0,0,0.06) 4px
    );
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION  (unchanged)
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
# HELPERS  (unchanged)
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


# ─────────────────────────────────────────────────────────────────────────────
# CHART STYLING — Bloomberg Terminal
# ─────────────────────────────────────────────────────────────────────────────
def style_fig(fig):
    """Apply Bloomberg Terminal style to a matplotlib figure."""
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(AX)
        ax.tick_params(colors=GRAY, labelsize=7.5, which="both")
        for sp in ax.spines.values():
            sp.set_color(BD)
            sp.set_linewidth(0.6)
        ax.yaxis.label.set_color(GRAY)
        ax.xaxis.label.set_color(GRAY)
        ax.title.set_color(TEXT)
        ax.title.set_fontsize(9)
        ax.title.set_fontweight("bold")
        ax.title.set_fontfamily("monospace")
        # Subtle grid
        ax.grid(True, color="#111111", linewidth=0.4, alpha=0.8, linestyle="-")
        ax.set_axisbelow(True)
        # Monospace tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("monospace")
            label.set_fontsize(7)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE  (unchanged)
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
# CHART BUILDERS — Bloomberg Terminal Style
# ─────────────────────────────────────────────────────────────────────────────
def chart_actual_vs_predicted(res, ticker):
    dt   = res["dates_test"]
    act  = res["prices_test"]
    pred = res["preds_inr"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG)

    ax1 = axes[0]
    ax1.set_facecolor(AX)
    ax1.plot(dt, act,  color=CYAN,  lw=1.6, label="ACTUAL", alpha=0.95)
    ax1.plot(dt, pred, color=AMBER, lw=1.2, ls="--", alpha=0.85, label="MODEL OUTPUT")
    ax1.fill_between(dt, act, pred,
                     where=pred >= act, alpha=0.08, color=GREEN)
    ax1.fill_between(dt, act, pred,
                     where=pred <  act, alpha=0.08, color=RED)
    ax1.set_ylabel("PRICE (INR)", fontsize=7.5, color=GRAY, labelpad=8)
    info_str = (f"MAE ₹{res['mae']:.0f}   RMSE ₹{res['rmse']:.0f}   "
                f"MAPE {res['mape']:.2f}%   R² {res['r2']:.4f}   ACC {res['acc']:.1f}%")
    ax1.text(0.01, 0.97, info_str, transform=ax1.transAxes,
             color=ORANGE, fontsize=7, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="square,pad=0.4", facecolor="#080808",
                       edgecolor="#1C1C1C", alpha=0.95))
    ax1.set_title("ACTUAL vs MODEL OUTPUT  ·  TEST SET", fontsize=9,
                  color=TEXT, fontweight="bold", loc="left", pad=10)
    ax1.legend(facecolor="#0A0A0A", edgecolor=BD, labelcolor=TEXT,
               fontsize=7.5, loc="upper right", framealpha=0.95,
               prop={"family": "monospace"})
    for sp in ax1.spines.values():
        sp.set_color(BD); sp.set_linewidth(0.6)

    ax2 = axes[1]
    ax2.set_facecolor(AX)
    n = min(90, len(dt))
    ax2.plot(dt[-n:], act[-n:],  color=CYAN,  lw=2,   label="ACTUAL")
    ax2.plot(dt[-n:], pred[-n:], color=AMBER, lw=1.6, ls="--", label="PREDICTED")
    ax2.fill_between(dt[-n:], act[-n:], pred[-n:],
                     where=pred[-n:] >= act[-n:], alpha=0.1, color=GREEN)
    ax2.fill_between(dt[-n:], act[-n:], pred[-n:],
                     where=pred[-n:] <  act[-n:], alpha=0.1, color=RED)
    ax2.set_ylabel("PRICE (INR)", fontsize=7.5, color=GRAY, labelpad=8)
    ax2.set_title("ZOOM · LAST 90 TRADING DAYS", fontsize=9,
                  color=TEXT, fontweight="bold", loc="left", pad=10)
    ax2.legend(facecolor="#0A0A0A", edgecolor=BD, labelcolor=TEXT,
               fontsize=7.5, prop={"family": "monospace"})
    for sp in ax2.spines.values():
        sp.set_color(BD); sp.set_linewidth(0.6)

    style_fig(fig)
    plt.tight_layout(pad=1.5)
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
    ax1 = axes[0]
    ax1.set_facecolor(AX)
    ax1.plot(hist_dates, hist_prices,
             color=CYAN, lw=1.8, label=f"HIST ({hist_n}D)")
    ax1.plot(bridge_d, bridge_p,
             color=ORANGE, lw=2.2, ls="--", marker="o",
             markersize=5, markerfacecolor=ORANGE,
             markeredgecolor="#000", markeredgewidth=0.8,
             label="7D FORECAST", zorder=5)
    if fc_dates:
        ax1.axvspan(fc_dates[0], fc_dates[-1], alpha=0.05, color=ORANGE)
    ax1.axvline(hist_dates[-1], color=TEXT, lw=0.8, ls=":", alpha=0.35)
    ax1.annotate(f"₹{last_price:,.0f}",
                 xy=(hist_dates[-1], last_price),
                 xytext=(-65, 14), textcoords="offset points",
                 color=TEXT, fontsize=7, fontfamily="monospace",
                 arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.7))
    ax1.annotate(f"₹{fc_prices[-1]:,.0f}",
                 xy=(fc_dates[-1], fc_prices[-1]),
                 xytext=(8, -18), textcoords="offset points",
                 color=ORANGE, fontsize=8, fontweight="bold", fontfamily="monospace",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    ax1.set_ylabel("PRICE (INR)", fontsize=7.5, color=GRAY, labelpad=8)
    ax1.set_title(f"{bank_name.upper()}  ·  7-DAY PRICE FORECAST",
                  fontsize=9, color=TEXT, fontweight="bold", loc="left", pad=10)
    ax1.legend(facecolor="#0A0A0A", edgecolor=BD, labelcolor=TEXT,
               fontsize=7.5, prop={"family": "monospace"})
    for sp in ax1.spines.values():
        sp.set_color(BD); sp.set_linewidth(0.6)

    # Right: bar chart by day
    ax2 = axes[1]
    ax2.set_facecolor(AX)
    day_labels = [f"D{i+1}\n{d.strftime('%d%b')}" for i, d in enumerate(fc_dates)]
    bar_colors = [GREEN if p >= last_price else RED for p in fc_prices]
    bars       = ax2.bar(day_labels, fc_prices, color=bar_colors,
                         alpha=0.80, edgecolor=BD, linewidth=0.5, width=0.55)
    ax2.axhline(last_price, color=CYAN, lw=1.2, ls="--",
                alpha=0.7, label=f"NOW ₹{last_price:,.0f}")
    ymin = min(fc_prices) * 0.998
    ymax = max(fc_prices) * 1.002
    ax2.set_ylim(ymin, ymax)
    for bar, price in zip(bars, fc_prices):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (ymax - ymin) * 0.003,
                 f"₹{price:,.0f}", ha="center",
                 color=TEXT, fontsize=7, fontfamily="monospace")
    ax2.set_title("DAY-BY-DAY PROJECTION",
                  fontsize=9, color=TEXT, fontweight="bold", loc="left", pad=10)
    ax2.legend(facecolor="#0A0A0A", edgecolor=BD, labelcolor=TEXT,
               fontsize=7.5, prop={"family": "monospace"})
    for sp in ax2.spines.values():
        sp.set_color(BD); sp.set_linewidth(0.6)

    style_fig(fig)
    plt.tight_layout(pad=1.5)
    return fig


def chart_technicals(df, ticker):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=BG,
                             gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    n   = min(504, len(df))
    sub = df.iloc[-n:]

    # Price + BBs + MAs
    ax1 = axes[0]
    ax1.set_facecolor(AX)
    ax1.plot(sub.index, sub["Close"],    color=CYAN,   lw=1.8, label="CLOSE", zorder=4)
    ax1.plot(sub.index, sub["ma_20"],    color=ORANGE, lw=0.9, ls="--", alpha=0.75, label="MA20")
    ax1.plot(sub.index, sub["ma_50"],    color=GREEN,  lw=0.9, ls="--", alpha=0.75, label="MA50")
    ax1.plot(sub.index, sub["bb_upper"], color=GRAY,   lw=0.7, ls=":",  alpha=0.55, label="BB+")
    ax1.plot(sub.index, sub["bb_lower"], color=GRAY,   lw=0.7, ls=":",  alpha=0.55, label="BB-")
    ax1.fill_between(sub.index, sub["bb_upper"], sub["bb_lower"],
                     alpha=0.04, color=CYAN)
    ax1.set_ylabel("PRICE (INR)", fontsize=7.5, color=GRAY, labelpad=8)
    ax1.set_title("PRICE · BOLLINGER BANDS · MOVING AVERAGES",
                  fontsize=9, color=TEXT, fontweight="bold", loc="left", pad=10)
    ax1.legend(facecolor="#0A0A0A", edgecolor=BD, labelcolor=TEXT,
               fontsize=7, ncol=3, prop={"family": "monospace"})
    for sp in ax1.spines.values():
        sp.set_color(BD); sp.set_linewidth(0.6)

    # RSI
    ax2 = axes[1]
    ax2.set_facecolor(AX)
    ax2.plot(sub.index, sub["rsi"], color=CYAN, lw=1.2)
    ax2.axhline(70, color=RED,   lw=0.8, ls="--", alpha=0.7, label="OB 70")
    ax2.axhline(30, color=GREEN, lw=0.8, ls="--", alpha=0.7, label="OS 30")
    ax2.axhline(50, color=GRAY,  lw=0.5, ls=":",  alpha=0.4)
    ax2.fill_between(sub.index, sub["rsi"], 70,
                     where=sub["rsi"] >= 70, alpha=0.12, color=RED)
    ax2.fill_between(sub.index, sub["rsi"], 30,
                     where=sub["rsi"] <= 30, alpha=0.12, color=GREEN)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", fontsize=7.5, color=GRAY, labelpad=8)
    ax2.legend(facecolor="#0A0A0A", edgecolor=BD, labelcolor=TEXT,
               fontsize=7, prop={"family": "monospace"})
    for sp in ax2.spines.values():
        sp.set_color(BD); sp.set_linewidth(0.6)

    # MACD
    ax3 = axes[2]
    ax3.set_facecolor(AX)
    ax3.plot(sub.index, sub["macd"],     color=CYAN,  lw=1.2, label="MACD")
    ax3.plot(sub.index, sub["macd_sig"], color=ORANGE, lw=1.2, ls="--", label="SIGNAL")
    bar_c = [GREEN if v >= 0 else RED for v in sub["macd_hist"]]
    ax3.bar(sub.index, sub["macd_hist"], color=bar_c, alpha=0.55,
            linewidth=0, label="HIST")
    ax3.axhline(0, color=GRAY, lw=0.5, ls=":", alpha=0.5)
    ax3.set_ylabel("MACD", fontsize=7.5, color=GRAY, labelpad=8)
    ax3.legend(facecolor="#0A0A0A", edgecolor=BD, labelcolor=TEXT,
               fontsize=7, prop={"family": "monospace"})
    for sp in ax3.spines.values():
        sp.set_color(BD); sp.set_linewidth(0.6)

    style_fig(fig)
    plt.tight_layout(pad=1.5)
    return fig


def chart_error(res):
    act  = res["prices_test"]
    pred = res["preds_inr"]
    dt   = res["dates_test"]
    err  = act - pred
    pct  = (err / act) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor=BG)

    ax1 = axes[0]
    ax1.set_facecolor(AX)
    ax1.plot(dt, pct, color=RED, lw=0.8, alpha=0.85)
    ax1.axhline(0,         color=TEXT, lw=0.8, ls="--", alpha=0.3)
    ax1.axhline(pct.mean(), color=ORANGE, lw=1.2, ls="--",
                label=f"MEAN {pct.mean():.2f}%")
    ax1.fill_between(dt, pct, 0, where=pct > 0, color=GREEN, alpha=0.1)
    ax1.fill_between(dt, pct, 0, where=pct < 0, color=RED,   alpha=0.1)
    ax1.set_ylabel("ERROR (%)", fontsize=7.5, color=GRAY, labelpad=8)
    ax1.set_title("PREDICTION ERROR % · TIME SERIES",
                  fontsize=9, color=TEXT, fontweight="bold", loc="left", pad=10)
    ax1.legend(facecolor="#0A0A0A", edgecolor=BD, labelcolor=TEXT,
               fontsize=7.5, prop={"family": "monospace"})
    for sp in ax1.spines.values():
        sp.set_color(BD); sp.set_linewidth(0.6)

    ax2 = axes[1]
    ax2.set_facecolor(AX)
    ax2.hist(pct, bins=60, color=CYAN, alpha=0.70, edgecolor="none", density=True)
    ax2.axvline(0,          color=TEXT,  lw=0.8, ls="--", alpha=0.4)
    ax2.axvline(pct.mean(), color=ORANGE, lw=1.4, ls="--",
                label=f"MEAN {pct.mean():.2f}%")
    ax2.set_xlabel("ERROR (%)", fontsize=7.5, color=GRAY, labelpad=8)
    ax2.set_ylabel("DENSITY",   fontsize=7.5, color=GRAY, labelpad=8)
    ax2.set_title("ERROR DISTRIBUTION",
                  fontsize=9, color=TEXT, fontweight="bold", loc="left", pad=10)
    ax2.legend(facecolor="#0A0A0A", edgecolor=BD, labelcolor=TEXT,
               fontsize=7.5, prop={"family": "monospace"})
    for sp in ax2.spines.values():
        sp.set_color(BD); sp.set_linewidth(0.6)

    style_fig(fig)
    plt.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE MODEL (retrain)
# ─────────────────────────────────────────────────────────────────────────────
def retrain_bank(ticker, bank_name, progress_bar, status_text):
    from torch.utils.data import Dataset, DataLoader

    class SeqDS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X)
            self.y = torch.tensor(y)
        def __len__(self):         return len(self.X)
        def __getitem__(self, i):  return self.X[i], self.y[i]

    status_text.text("▸ DOWNLOADING LATEST MARKET DATA...")
    bank_df  = fetch_data(ticker)
    nifty_df = fetch_data(NIFTY_BANK)
    common   = bank_df.index.intersection(nifty_df.index)
    df       = add_features(bank_df.loc[common], nifty_df.loc[common])

    status_text.text("▸ PREPARING SEQUENCE TENSORS...")
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
        status_text.text(f"▸ EPOCH {ep:02d}/{EPOCHS}  ·  VAL_LOSS {avg_vl:.5f}")

    model.load_state_dict(best_st)

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

    load_model.clear()
    fetch_data.clear()
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Bloomberg Terminal Panel
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Top status bar
    st.markdown(f"""
    <div style='background:#050505; border-bottom:1px solid #1C1C1C;
                padding:12px 16px 10px 16px; margin:-1rem -1rem 0 -1rem;'>
        <div style='display:flex; align-items:center; gap:8px; margin-bottom:4px;'>
            <span style='color:#F7921D; font-size:1rem; font-weight:700;
                         letter-spacing:0.05em;'>INDIABANK</span>
            <span style='color:#333; font-size:0.65rem;'>|</span>
            <span style='color:#555; font-size:0.6rem; letter-spacing:0.12em;'>LSTM FORECASTER</span>
        </div>
        <div style='display:flex; align-items:center; gap:6px;'>
            <span class='live-dot' style='width:6px; height:6px; border-radius:50%;
                  background:#00C853; display:inline-block;'></span>
            <span style='color:#00C853; font-size:0.6rem; letter-spacing:0.1em;'>SYSTEM ONLINE</span>
            <span style='color:#333; margin-left:auto; font-size:0.58rem;'>
                {datetime.now().strftime('%H:%M:%S')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Bank selector
    st.markdown("""
    <div style='font-size:0.6rem; color:#555; letter-spacing:0.14em;
                text-transform:uppercase; margin-bottom:6px; padding:0 2px;'>
        ▸ INSTRUMENT SELECT
    </div>
    """, unsafe_allow_html=True)

    bank_options = list(BANKS.items())
    names        = [n for _, n in bank_options]
    tickers      = [t for t, _ in bank_options]
    selected_name   = st.selectbox("", names, label_visibility="collapsed")
    selected_ticker = tickers[names.index(selected_name)]

    # Model status card
    exists = model_exists(selected_ticker)
    if exists:
        with open(os.path.join(MODELS_DIR, selected_ticker, "meta.json")) as f:
            cached_meta = json.load(f)
        acc_color = GREEN if cached_meta.get("accuracy", 0) >= 90 else AMBER
        st.markdown(f"""
        <div style='background:#050505; border:1px solid #1C1C1C;
                    border-left:2px solid #00C853;
                    padding:10px 12px; margin:10px 0;'>
            <div style='color:#00C853; font-size:0.6rem; font-weight:700;
                        letter-spacing:0.14em; margin-bottom:8px;'>
                ■ MODEL READY
            </div>
            <table style='width:100%; border-collapse:collapse;
                          font-size:0.62rem; color:#777;'>
                <tr>
                    <td style='color:#444; padding:2px 0;'>TRAINED</td>
                    <td style='color:#AAAAAA; text-align:right;'>
                        {cached_meta.get("trained_on","—")}</td>
                </tr>
                <tr>
                    <td style='color:#444; padding:2px 0;'>DATA RANGE</td>
                    <td style='color:#AAAAAA; text-align:right; font-size:0.58rem;'>
                        {cached_meta.get("data_start","—")}<br>→ {cached_meta.get("data_end","—")}</td>
                </tr>
                <tr>
                    <td style='color:#444; padding:2px 0;'>ACCURACY</td>
                    <td style='color:{acc_color}; text-align:right; font-weight:700;'>
                        {cached_meta.get("accuracy","—")}%</td>
                </tr>
                <tr>
                    <td style='color:#444; padding:2px 0;'>R² SCORE</td>
                    <td style='color:#F7921D; text-align:right; font-weight:600;'>
                        {cached_meta.get("r2","—")}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#050505; border:1px solid #1C1C1C;
                    border-left:2px solid #FF3B3B;
                    padding:10px 12px; margin:10px 0;'>
            <div style='color:#FF3B3B; font-size:0.6rem; font-weight:700;
                        letter-spacing:0.14em;'>
                □ NO MODEL FOUND
            </div>
            <div style='color:#444; font-size:0.6rem; margin-top:6px;'>
                RUN train_all_banks.py
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Action buttons
    predict_btn = st.button("▶  EXECUTE FORECAST",
                            use_container_width=True, disabled=not exists)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    update_btn  = st.button("⟳  SYNC + RETRAIN MODEL",
                            use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Model config panel
    st.markdown(f"""
    <div style='border-top:1px solid #1C1C1C; padding-top:14px;'>
        <div style='font-size:0.58rem; color:#444; letter-spacing:0.12em;
                    margin-bottom:8px; text-transform:uppercase;'>
            ▸ MODEL PARAMETERS
        </div>
        <table style='width:100%; border-collapse:collapse; font-size:0.6rem;'>
            <tr>
                <td style='color:#444; padding:2px 0;'>ARCHITECTURE</td>
                <td style='color:#777; text-align:right;'>LSTM ×{N_LAYERS}</td>
            </tr>
            <tr>
                <td style='color:#444; padding:2px 0;'>HIDDEN UNITS</td>
                <td style='color:#777; text-align:right;'>{HIDDEN}</td>
            </tr>
            <tr>
                <td style='color:#444; padding:2px 0;'>SEQ LENGTH</td>
                <td style='color:#777; text-align:right;'>{SEQ_LEN}D</td>
            </tr>
            <tr>
                <td style='color:#444; padding:2px 0;'>FEATURES</td>
                <td style='color:#777; text-align:right;'>{N_FEATURES}</td>
            </tr>
            <tr>
                <td style='color:#444; padding:2px 0;'>FORECAST</td>
                <td style='color:#F7921D; text-align:right; font-weight:700;'>
                    {FORECAST_DAYS}D AHEAD</td>
            </tr>
            <tr>
                <td style='color:#444; padding:2px 0;'>DEVICE</td>
                <td style='color:#777; text-align:right; font-size:0.58rem;'>
                    {str(DEVICE).upper()}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style='position:fixed; bottom:0; left:0; width:260px;
                background:#030303; border-top:1px solid #1A1A1A;
                padding:8px 16px; font-size:0.55rem; color:#2A2A2A;'>
        BTECH CAPSTONE  ·  INDIAN BANKING SECTOR  ·  LSTM
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

# ── Bloomberg-style top header bar ──────────────────────────────────────────
st.markdown(f"""
<div style='background:#000; border-bottom:1px solid #1C1C1C;
            padding:10px 0 12px 0; margin-bottom:20px; position:relative;'>
    <!-- Instrument name bar -->
    <div style='display:flex; align-items:baseline; gap:14px; margin-bottom:6px;'>
        <span style='font-size:1.45rem; font-weight:700; color:#F7F7F7;
                     letter-spacing:-0.02em;'>{selected_name.upper()}</span>
        <span style='font-size:0.75rem; color:#444; font-family:IBM Plex Mono;'>
            {selected_ticker}</span>
        <span style='font-size:0.6rem; color:#2A2A2A; margin-left:auto;
                     font-variant-numeric:tabular-nums;'>
            {datetime.now().strftime('%a %d %b %Y  %H:%M:%S IST')}</span>
    </div>
    <!-- Info strip -->
    <div style='display:flex; align-items:center; gap:0;'>
        <span style='background:#0D0D0D; border:1px solid #1C1C1C;
                     color:#555; font-size:0.6rem; padding:3px 10px;
                     letter-spacing:0.1em;'>NSE</span>
        <span style='background:#0D0D0D; border:1px solid #1C1C1C; border-left:none;
                     color:#555; font-size:0.6rem; padding:3px 10px;
                     letter-spacing:0.1em;'>BANKING SECTOR</span>
        <span style='background:#0D0D0D; border:1px solid #1C1C1C; border-left:none;
                     color:#555; font-size:0.6rem; padding:3px 10px;
                     letter-spacing:0.1em;'>LSTM FORECAST · 20Y DATA</span>
        <span style='background:#F7921D18; border:1px solid #F7921D44; border-left:none;
                     color:#F7921D; font-size:0.6rem; padding:3px 10px;
                     letter-spacing:0.1em; font-weight:600;'>7-DAY HORIZON</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── UPDATE MODEL flow ─────────────────────────────────────────────────────────
if update_btn:
    st.markdown("""
    <div style='background:#050505; border:1px solid #1C1C1C; border-left:2px solid #F7921D;
                padding:10px 16px; margin-bottom:12px; font-size:0.72rem; color:#F7921D;
                letter-spacing:0.08em;'>
        ▸ INITIATING MODEL RETRAIN SEQUENCE
    </div>
    """, unsafe_allow_html=True)
    prog  = st.progress(0)
    stat  = st.empty()
    try:
        new_meta = retrain_bank(selected_ticker, selected_name, prog, stat)
        prog.progress(1.0)
        stat.empty()
        st.success(
            f"▸ RETRAIN COMPLETE  ·  "
            f"ACC {new_meta['accuracy']:.1f}%  ·  "
            f"DATA THROUGH {new_meta['data_end']}"
        )
    except Exception as e:
        stat.empty()
        st.error(f"▸ RETRAIN FAILED  ·  {e}")
    st.rerun()


# ── PREDICT flow ──────────────────────────────────────────────────────────────
if predict_btn and exists:
    with st.spinner("LOADING MODEL · RUNNING INFERENCE..."):
        model, scaler, price_scaler, meta = load_model(selected_ticker)
        res = run_prediction(selected_ticker, model, scaler, price_scaler, meta)

    # ── Bloomberg-style ticker strip ─────────────────────────────────────
    last_close = float(res["df"]["Close"].iloc[-1])
    prev_close = float(res["df"]["Close"].iloc[-2])
    day_chg    = last_close - prev_close
    day_pct    = day_chg / prev_close * 100
    week_ret   = (last_close / float(res["df"]["Close"].iloc[-6]) - 1) * 100
    month_ret  = (last_close / float(res["df"]["Close"].iloc[-22]) - 1) * 100
    year_ret   = (last_close / float(res["df"]["Close"].iloc[-252]) - 1) * 100
    fc7_chg    = res["fc_prices"][-1] - last_close
    fc7_pct    = fc7_chg / last_close * 100

    def _color(v):
        return GREEN if v >= 0 else RED
    def _arrow(v):
        return "▲" if v >= 0 else "▼"

    st.markdown(f"""
    <div style='display:grid; grid-template-columns:repeat(6,1fr);
                gap:1px; background:#111; border:1px solid #1C1C1C;
                margin-bottom:16px;'>
        <!-- Card 1 -->
        <div style='background:#000; padding:12px 14px;'>
            <div style='color:#555; font-size:0.58rem; letter-spacing:0.12em;
                        text-transform:uppercase; margin-bottom:6px;'>LAST PRICE</div>
            <div style='color:#F7F7F7; font-size:1.25rem; font-weight:700;
                        letter-spacing:-0.02em; font-variant-numeric:tabular-nums;'>
                ₹{last_close:,.2f}</div>
            <div style='color:{_color(day_chg)}; font-size:0.68rem; margin-top:4px;
                        font-variant-numeric:tabular-nums;'>
                {_arrow(day_chg)} {day_chg:+.2f} ({day_pct:+.2f}%)</div>
        </div>
        <!-- Card 2 -->
        <div style='background:#000; padding:12px 14px; border-left:1px solid #111;'>
            <div style='color:#555; font-size:0.58rem; letter-spacing:0.12em;
                        text-transform:uppercase; margin-bottom:6px;'>7D FORECAST</div>
            <div style='color:#F7921D; font-size:1.25rem; font-weight:700;
                        letter-spacing:-0.02em; font-variant-numeric:tabular-nums;'>
                ₹{res["fc_prices"][-1]:,.2f}</div>
            <div style='color:{_color(fc7_chg)}; font-size:0.68rem; margin-top:4px;
                        font-variant-numeric:tabular-nums;'>
                {_arrow(fc7_chg)} {fc7_chg:+.2f} ({fc7_pct:+.2f}%)</div>
        </div>
        <!-- Card 3 -->
        <div style='background:#000; padding:12px 14px; border-left:1px solid #111;'>
            <div style='color:#555; font-size:0.58rem; letter-spacing:0.12em;
                        text-transform:uppercase; margin-bottom:6px;'>1W RETURN</div>
            <div style='color:{_color(week_ret)}; font-size:1.25rem; font-weight:700;
                        letter-spacing:-0.02em; font-variant-numeric:tabular-nums;'>
                {_arrow(week_ret)}{week_ret:+.2f}%</div>
            <div style='color:#333; font-size:0.6rem; margin-top:4px;'>TRAILING</div>
        </div>
        <!-- Card 4 -->
        <div style='background:#000; padding:12px 14px; border-left:1px solid #111;'>
            <div style='color:#555; font-size:0.58rem; letter-spacing:0.12em;
                        text-transform:uppercase; margin-bottom:6px;'>1M RETURN</div>
            <div style='color:{_color(month_ret)}; font-size:1.25rem; font-weight:700;
                        letter-spacing:-0.02em; font-variant-numeric:tabular-nums;'>
                {_arrow(month_ret)}{month_ret:+.2f}%</div>
            <div style='color:#333; font-size:0.6rem; margin-top:4px;'>TRAILING</div>
        </div>
        <!-- Card 5 -->
        <div style='background:#000; padding:12px 14px; border-left:1px solid #111;'>
            <div style='color:#555; font-size:0.58rem; letter-spacing:0.12em;
                        text-transform:uppercase; margin-bottom:6px;'>1Y RETURN</div>
            <div style='color:{_color(year_ret)}; font-size:1.25rem; font-weight:700;
                        letter-spacing:-0.02em; font-variant-numeric:tabular-nums;'>
                {_arrow(year_ret)}{year_ret:+.2f}%</div>
            <div style='color:#333; font-size:0.6rem; margin-top:4px;'>TRAILING</div>
        </div>
        <!-- Card 6 -->
        <div style='background:#000; padding:12px 14px; border-left:1px solid #111;'>
            <div style='color:#555; font-size:0.58rem; letter-spacing:0.12em;
                        text-transform:uppercase; margin-bottom:6px;'>MODEL ACC</div>
            <div style='color:#00C853; font-size:1.25rem; font-weight:700;
                        letter-spacing:-0.02em; font-variant-numeric:tabular-nums;'>
                {res["acc"]:.1f}%</div>
            <div style='color:#444; font-size:0.6rem; margin-top:4px;'>
                R² {res["r2"]:.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 7-day forecast table ─────────────────────────────────────────────
    st.markdown("""
    <div style='font-size:0.62rem; color:#555; letter-spacing:0.14em;
                text-transform:uppercase; margin-bottom:8px;'>
        ▸ 7-DAY PRICE PROJECTION TABLE
    </div>
    """, unsafe_allow_html=True)

    # Build styled HTML table
    prev = last_close
    rows_html = ""
    for i, (d, p) in enumerate(zip(res["fc_dates"], res["fc_prices"])):
        chg   = p - prev
        pct_v = chg / prev * 100
        sig   = "▲ LONG" if chg > 0 else "▼ SHORT"
        sig_c = GREEN if chg > 0 else RED
        row_bg = "#050505" if i % 2 == 0 else "#020202"
        rows_html += f"""
        <tr style='background:{row_bg};'>
            <td style='padding:7px 12px; color:#555; font-size:0.62rem;'>D{i+1}</td>
            <td style='padding:7px 12px; color:#888; font-size:0.62rem;'>
                {d.strftime("%a  %d %b %Y")}</td>
            <td style='padding:7px 12px; color:#F7F7F7; font-weight:600;
                       font-variant-numeric:tabular-nums;'>₹{p:,.2f}</td>
            <td style='padding:7px 12px; color:{_color(chg)};
                       font-variant-numeric:tabular-nums;'>{chg:+.2f}</td>
            <td style='padding:7px 12px; color:{_color(pct_v)};
                       font-variant-numeric:tabular-nums;'>{pct_v:+.2f}%</td>
            <td style='padding:7px 12px; color:{sig_c}; font-weight:700;
                       font-size:0.65rem; letter-spacing:0.06em;'>{sig}</td>
        </tr>"""
        prev = p

    st.markdown(f"""
    <div style='border:1px solid #1C1C1C; overflow:hidden; margin-bottom:20px;'>
        <table style='width:100%; border-collapse:collapse;
                      font-family:IBM Plex Mono, monospace;'>
            <thead>
                <tr style='background:#080808; border-bottom:1px solid #1C1C1C;'>
                    <th style='padding:7px 12px; color:#444; font-size:0.58rem;
                               letter-spacing:0.12em; text-align:left; font-weight:500;'>DAY</th>
                    <th style='padding:7px 12px; color:#444; font-size:0.58rem;
                               letter-spacing:0.12em; text-align:left; font-weight:500;'>DATE</th>
                    <th style='padding:7px 12px; color:#444; font-size:0.58rem;
                               letter-spacing:0.12em; text-align:left; font-weight:500;'>PRICE</th>
                    <th style='padding:7px 12px; color:#444; font-size:0.58rem;
                               letter-spacing:0.12em; text-align:left; font-weight:500;'>Δ INR</th>
                    <th style='padding:7px 12px; color:#444; font-size:0.58rem;
                               letter-spacing:0.12em; text-align:left; font-weight:500;'>Δ %</th>
                    <th style='padding:7px 12px; color:#444; font-size:0.58rem;
                               letter-spacing:0.12em; text-align:left; font-weight:500;'>SIGNAL</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ────────────────────────────────────────────────────────────
    st.markdown("<div style='border-top:1px solid #1C1C1C; margin-bottom:0;'></div>",
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "ACTUAL vs PREDICTED",
        "FORECAST CHART",
        "TECHNICAL INDICATORS",
        "ERROR ANALYSIS",
    ])

    with tab1:
        # Metrics strip
        st.markdown(f"""
        <div style='display:grid; grid-template-columns:repeat(5,1fr);
                    gap:1px; background:#111; border:1px solid #1C1C1C;
                    margin-bottom:16px;'>
            <div style='background:#000; padding:10px 14px;'>
                <div style='color:#444; font-size:0.58rem; letter-spacing:0.12em;
                            text-transform:uppercase; margin-bottom:4px;'>MAE</div>
                <div style='color:#00B8FF; font-size:1.1rem; font-weight:700;
                            font-variant-numeric:tabular-nums;'>₹{res["mae"]:.2f}</div>
            </div>
            <div style='background:#000; padding:10px 14px; border-left:1px solid #111;'>
                <div style='color:#444; font-size:0.58rem; letter-spacing:0.12em;
                            text-transform:uppercase; margin-bottom:4px;'>RMSE</div>
                <div style='color:#FFB800; font-size:1.1rem; font-weight:700;
                            font-variant-numeric:tabular-nums;'>₹{res["rmse"]:.2f}</div>
            </div>
            <div style='background:#000; padding:10px 14px; border-left:1px solid #111;'>
                <div style='color:#444; font-size:0.58rem; letter-spacing:0.12em;
                            text-transform:uppercase; margin-bottom:4px;'>MAPE</div>
                <div style='color:#FF3B3B; font-size:1.1rem; font-weight:700;
                            font-variant-numeric:tabular-nums;'>{res["mape"]:.2f}%</div>
            </div>
            <div style='background:#000; padding:10px 14px; border-left:1px solid #111;'>
                <div style='color:#444; font-size:0.58rem; letter-spacing:0.12em;
                            text-transform:uppercase; margin-bottom:4px;'>R² SCORE</div>
                <div style='color:#00C853; font-size:1.1rem; font-weight:700;
                            font-variant-numeric:tabular-nums;'>{res["r2"]:.4f}</div>
            </div>
            <div style='background:#000; padding:10px 14px; border-left:1px solid #111;'>
                <div style='color:#444; font-size:0.58rem; letter-spacing:0.12em;
                            text-transform:uppercase; margin-bottom:4px;'>ACCURACY</div>
                <div style='color:#00C853; font-size:1.1rem; font-weight:700;
                            font-variant-numeric:tabular-nums;'>{res["acc"]:.1f}%</div>
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
    # ── Landing idle state ───────────────────────────────────────────────
    st.markdown("""
    <div style='border:1px solid #1C1C1C; background:#030303;
                padding:60px 40px; text-align:center; margin-bottom:24px;'>
        <div style='color:#1C1C1C; font-size:3.5rem; font-family:monospace;
                    letter-spacing:-0.04em; margin-bottom:20px;'>LSTM</div>
        <div style='color:#333; font-size:0.72rem; letter-spacing:0.2em;
                    text-transform:uppercase; margin-bottom:8px;'>
            SELECT INSTRUMENT · EXECUTE FORECAST
        </div>
        <div style='color:#1C1C1C; font-size:0.6rem; letter-spacing:0.14em;'>
            7-DAY PRICE PROJECTION · 20-YEAR TRAINING DATA · NIFTY BANK CORRELATED
        </div>
        <div style='margin-top:24px; color:#F7921D; font-size:0.65rem;
                    letter-spacing:0.18em; opacity:0.6;'>
            ▸ PRESS  EXECUTE FORECAST  TO BEGIN
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bank status grid
    st.markdown("""
    <div style='font-size:0.6rem; color:#444; letter-spacing:0.14em;
                text-transform:uppercase; margin-bottom:10px;'>
        ▸ MODEL REGISTRY  ·  ALL INSTRUMENTS
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    for i, (t, n) in enumerate(BANKS.items()):
        with cols[i % 4]:
            ok = model_exists(t)
            if ok:
                with open(os.path.join(MODELS_DIR, t, "meta.json")) as f:
                    m = json.load(f)
                acc = m.get("accuracy", 0)
                acc_c = GREEN if acc >= 90 else AMBER
                st.markdown(f"""
                <div style='background:#030303; border:1px solid #1C1C1C;
                            border-top:2px solid #00C853;
                            padding:10px 12px; margin-bottom:8px;'>
                    <div style='color:#00C853; font-size:0.58rem; font-weight:700;
                                letter-spacing:0.12em; margin-bottom:6px;'>
                        ■ {n.upper()}</div>
                    <div style='font-size:0.6rem; color:#444; line-height:1.8;'>
                        <span style='color:{acc_c}; font-weight:700;'>
                            ACC {acc}%</span>
                        &nbsp;&nbsp;
                        <span style='color:#F7921D;'>R² {m.get("r2","—")}</span><br>
                        <span style='color:#333;'>₹{m.get("last_close","—")}
                        &nbsp;|&nbsp; {m.get("data_end","—")}</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:#030303; border:1px solid #1C1C1C;
                            border-top:2px solid #2A2A2A;
                            padding:10px 12px; margin-bottom:8px;'>
                    <div style='color:#333; font-size:0.58rem; font-weight:700;
                                letter-spacing:0.12em; margin-bottom:6px;'>
                        □ {n.upper()}</div>
                    <div style='color:#222; font-size:0.6rem;'>NOT TRAINED</div>
                </div>""", unsafe_allow_html=True)


elif not exists:
    st.markdown(f"""
    <div style='background:#030303; border:1px solid #1C1C1C;
                border-left:3px solid #FF3B3B;
                padding:16px 20px; font-family:IBM Plex Mono, monospace;'>
        <div style='color:#FF3B3B; font-size:0.65rem; font-weight:700;
                    letter-spacing:0.12em; margin-bottom:6px;'>
            ■ MODEL NOT FOUND  ·  {selected_ticker}
        </div>
        <div style='color:#444; font-size:0.65rem;'>
            RUN:  python train_all_banks.py
        </div>
    </div>
    """, unsafe_allow_html=True)
