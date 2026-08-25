# AI Crypto Trading Bot

An original Python trading-bot template inspired by the public outline of **AI Trading Bot with Python: Machine Learning & Backtest 2026** by The Inspiring Trader.

It does not include or reproduce paid Udemy course source files. If you add your purchased course template files to this workspace, the project can be adapted around them.

## What This Includes

- 15-minute OHLCV crypto data workflow
- Leakage-safe train/test split and feature scaling
- Technical indicators and stationarity-style return features
- Machine-learning direction prediction
- Backtesting with fees, slippage, stop loss, and position sizing
- Paper/live bot skeleton for exchange integration
- CLI commands for analysis, training, backtesting, and paper decisions

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Format

CSV files should contain:

```text
timestamp,open,high,low,close,volume
```

`timestamp` can be a datetime string or Unix timestamp. Example path:

```text
data/BTCUSDT_15m.csv
```

## Quick Start

```powershell
python -m trading_bot.cli analyze --csv data/BTCUSDT_15m.csv
python -m trading_bot.cli train --csv data/BTCUSDT_15m.csv --model-out models/btc_model.joblib
python -m trading_bot.cli backtest --csv data/BTCUSDT_15m.csv --model models/btc_model.joblib
python -m trading_bot.cli paper --csv data/BTCUSDT_15m.csv --model models/btc_model.joblib
```

## Safety

This is research software, not financial advice. Start with historical backtests and paper trading. Live trading requires exchange credentials, careful risk limits, and jurisdiction-specific compliance checks.

