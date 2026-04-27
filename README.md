# 📈 Indian Banking Sector — LSTM Stock Price Forecaster
### BTech Capstone Project

---

## Project Structure

```
project/
├── train_all_banks.py     ← Step 1: Train & save all 8 bank models
├── app.py                 ← Step 2: Launch Streamlit dashboard
├── requirements.txt       ← All dependencies
└── models/                ← Auto-created after training
    ├── HDFCBANK.NS/
    │   ├── model.pt
    │   ├── scaler.pkl
    │   ├── price_scaler.pkl
    │   └── meta.json
    ├── ICICIBANK.NS/
    └── ...
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all 8 bank models (run ONCE — takes ~30-60 min on CPU)
```bash
python train_all_banks.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

---

## Banks Covered

| # | Bank              | Ticker        |
|---|-------------------|---------------|
| 1 | HDFC Bank         | HDFCBANK.NS   |
| 2 | ICICI Bank        | ICICIBANK.NS  |
| 3 | State Bank of India | SBIN.NS     |
| 4 | Kotak Mahindra Bank | KOTAKBANK.NS|
| 5 | Axis Bank         | AXISBANK.NS   |
| 6 | IndusInd Bank     | INDUSINDBK.NS |
| 7 | Bank of Baroda    | BANKBARODA.NS |
| 8 | Punjab National Bank | PNB.NS     |

---

## Features Used (23 total)

| Category        | Features                                          |
|-----------------|---------------------------------------------------|
| Momentum        | Daily return, MA-10/20/50, distance from MAs      |
| Volatility      | 10-day & 20-day rolling std                       |
| RSI             | 14-day Relative Strength Index                    |
| MACD            | MACD line, Signal line, Histogram                 |
| Bollinger Bands | BB position, BB width                             |
| Volume          | Volume ratio vs 20-day average                    |
| Correlated Asset| NIFTY Bank daily return, NIFTY Bank MA distance   |
| Lag features    | Return lag 1, 2, 3, 5 days                        |
| Target          | Close price (scaled)                              |

---

## Model Architecture

```
Input (60 days × 23 features)
    ↓
LSTM Layer 1 (128 hidden units)
    ↓
LSTM Layer 2 (128 hidden units)  + Dropout(0.2)
    ↓
LayerNorm → Dropout
    ↓
FC(128 → 64) → ReLU → Dropout
    ↓
FC(64 → 1)
    ↓
Output: Next day's Close price
```

---

## Dashboard Features

- **Select any bank** from sidebar dropdown
- **Run Forecast** button — predicts next 7 trading days instantly
- **Update Model** button — downloads latest data from Yahoo Finance and retrains
- **4 analysis tabs:**
  - Actual vs Predicted price chart (full test set + 90-day zoom)
  - 7-day forecast chart (line + bar)
  - Technical indicators (Price + BBs + MAs + RSI + MACD)
  - Error analysis (error over time + distribution)
- **Accuracy metrics:** MAE, RMSE, MAPE, R², Accuracy %
- **Live metrics row:** Current price, 7-day forecast, weekly/monthly/yearly returns
