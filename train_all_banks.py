

import os, json, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
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
END_DATE    = datetime.today().strftime("%Y-%m-%d")

SEQ_LEN     = 60
TRAIN_SPLIT = 0.85
EPOCHS      = 60
BATCH_SIZE  = 64
LR          = 1e-3
HIDDEN      = 128
N_LAYERS    = 2
DROPOUT     = 0.2
MODELS_DIR  = "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FEATURE_COLS = [
    "ret", "ma_10", "ma_20", "ma_50",
    "dist_ma20", "dist_ma50", "dist_ma200",
    "vol_10", "vol_20",
    "rsi", "macd", "macd_sig", "macd_hist",
    "bb_pos", "bb_width",
    "vol_ratio",
    "nifty_ret", "nifty_dist",
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
    "Close",
]
TARGET_IDX = FEATURE_COLS.index("Close")
N_FEATURES = len(FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def fetch(ticker, start=START_DATE, end=END_DATE):
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    raw.dropna(inplace=True)
    return raw


def add_features(df, nifty):
    f = df.copy()
    c = f["Close"]
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


def build_sequences(data, seq_len, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self, n_features=N_FEATURES, hidden=HIDDEN,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.norm(out[:, -1]))).squeeze(1)


class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
    def __len__(self):         return len(self.X)
    def __getitem__(self, i):  return self.X[i], self.y[i]


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE BANK
# ─────────────────────────────────────────────────────────────────────────────
def train_bank(ticker, name, nifty_df):
    print(f"\n{'='*60}")
    print(f"  Training: {name} ({ticker})")
    print(f"{'='*60}")

    save_dir = os.path.join(MODELS_DIR, ticker)
    os.makedirs(save_dir, exist_ok=True)

    # Download & features
    bank_df = fetch(ticker)
    common  = bank_df.index.intersection(nifty_df.index)
    bank_df = bank_df.loc[common]
    nifty_a = nifty_df.loc[common]
    df      = add_features(bank_df, nifty_a)

    # Scale
    data_arr     = df[FEATURE_COLS].values
    scaler       = MinMaxScaler()
    data_sc      = scaler.fit_transform(data_arr)
    price_scaler = MinMaxScaler()
    price_scaler.fit(df[["Close"]].values)

    # Sequences
    X, y      = build_sequences(data_sc, SEQ_LEN, TARGET_IDX)
    N         = len(X)
    split     = int(N * TRAIN_SPLIT)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    dates_all  = df.index[SEQ_LEN:]
    prices_test = df["Close"].values[SEQ_LEN:][split:]

    print(f"  Rows: {len(df)} | Train: {split} | Test: {N-split}")

    tr_dl = DataLoader(SeqDS(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    te_dl = DataLoader(SeqDS(X_te, y_te), batch_size=BATCH_SIZE, shuffle=False)

    model   = StockLSTM().to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sch     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit    = nn.MSELoss()
    best_vl = float("inf")
    best_st = None
    tr_losses, va_losses = [], []

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in te_dl:
                vl += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        avg_tr = ep_loss / len(tr_dl)
        avg_vl = vl / len(te_dl)
        tr_losses.append(avg_tr)
        va_losses.append(avg_vl)
        sch.step(avg_vl)
        if avg_vl < best_vl:
            best_vl = avg_vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"  Ep {ep:3d}/{EPOCHS}  train={avg_tr:.5f}  val={avg_vl:.5f}")

    model.load_state_dict(best_st)

    # Evaluate
    model.eval()
    preds_sc = []
    with torch.no_grad():
        for xb, _ in DataLoader(SeqDS(X_te, y_te), batch_size=256):
            preds_sc.extend(model(xb.to(DEVICE)).cpu().numpy())
    preds_sc  = np.array(preds_sc, dtype=np.float32)
    preds_inr = price_scaler.inverse_transform(preds_sc.reshape(-1,1)).flatten()

    mae  = mean_absolute_error(prices_test, preds_inr)
    rmse = np.sqrt(mean_squared_error(prices_test, preds_inr))
    mape = np.mean(np.abs((prices_test - preds_inr) / prices_test)) * 100
    r2   = r2_score(prices_test, preds_inr)

    print(f"  MAE=₹{mae:.2f}  RMSE=₹{rmse:.2f}  MAPE={mape:.2f}%  R²={r2:.4f}")

    # Save everything
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, "price_scaler.pkl"), "wb") as f:
        pickle.dump(price_scaler, f)

    meta = {
        "ticker": ticker, "name": name,
        "trained_on": END_DATE,
        "data_start": str(df.index[0].date()),
        "data_end":   str(df.index[-1].date()),
        "train_rows": split, "test_rows": N - split,
        "mae": round(mae, 2), "rmse": round(rmse, 2),
        "mape": round(mape, 2), "r2": round(r2, 4),
        "accuracy": round(max(0, 100 - mape), 2),
        "last_close": round(float(df["Close"].iloc[-1]), 2),
        "train_losses": tr_losses,
        "val_losses":   va_losses,
        "feature_cols": FEATURE_COLS,
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved → {save_dir}/")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Indian Bank LSTM — Training All {len(BANKS)} Banks")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\nDownloading NIFTY Bank Index...")
    nifty_df = fetch(NIFTY_BANK)
    print(f"NIFTY rows: {len(nifty_df)}")

    results = {}
    for ticker, name in BANKS.items():
        try:
            meta = train_bank(ticker, name, nifty_df)
            results[ticker] = meta
        except Exception as e:
            print(f"  ERROR for {ticker}: {e}")

    # Save summary
    with open(os.path.join(MODELS_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE — Summary")
    print(f"{'='*60}")
    for ticker, m in results.items():
        print(f"  {m['name']:30s}  Acc={m['accuracy']:.1f}%  R²={m['r2']:.4f}")
    print(f"\n  All models saved in ./models/")
    print(f"  Now run:  streamlit run app.py")
