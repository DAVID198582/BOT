# AI Crypto Trading Bot

Research-oriented Python trading-bot template for crypto (15-minute OHLCV).

**Version 0.1.1** — critical backtest timing (look-ahead) fixed.

## What This Includes

- 15-minute OHLCV crypto data workflow
- Leakage-safe train/test split and feature scaling
- Technical indicators and return / volatility features
- Machine-learning direction prediction (Random Forest)
- Backtesting with fees, slippage, stop-loss, and position sizing
- **Correct signal execution lag**: signals generated on bar *t* are acted on at the open of bar *t+1*
- Paper-trading decision skeleton
- CLI for analysis, training, backtesting, and paper decisions

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt