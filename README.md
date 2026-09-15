# AI Crypto Trading Bot

A leakage-safe Python crypto research and spot-trading bot for OHLCV data. It can
analyze markets, train a random-forest direction model, backtest it, make a
one-off paper decision, or continuously consume completed candles from a
CCXT-supported exchange.

> Trading is risky. No model can guarantee profit. Live mode is deliberately
> disabled by default; test the strategy and exchange sandbox before using funds.

## Features

- Technical, momentum, volatility, volume, RSI, Bollinger, and ATR features
- Chronological train/test split with an sklearn model pipeline
- Backtesting with fees, slippage, stop-loss, and next-bar execution
- Live public market-data polling without API credentials
- Persistent dry-run balances and positions across restarts
- Exchange-native stop-loss/take-profit when supported, with client-side fallback
- ATR-aware risk sizing, trailing stops, break-even stops, and profit targets
- Daily-loss and consecutive-loss circuit breakers
- Partial-fill, fee, balance, and order-status reconciliation
- JSONL trade journal plus optional Telegram and email alerts
- Expanding-window validation that rejects weak models before live use
- Multi-symbol exposure, position-count, and correlation controls
- Local monitoring dashboard with a persistent emergency halt
- Spot-only execution: a negative signal exits a managed long; it never opens a short
- Duplicate-candle protection, exchange minimum checks, and explicit live-order opt-in

## Install

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

All commands are run from the repository root. Training now performs walk-forward
validation and refuses to save a weak model unless `--allow-weak-model` is passed.
The CSV must contain `timestamp,open,high,low,close,volume` columns:

```bash
python -m trading_bot.cli train \
  --csv data/btc_usdt_15m.csv \
  --model-out models/btc.joblib
```

Analyze and backtest it:

```bash
python -m trading_bot.cli analyze --csv data/btc_usdt_15m.csv
python -m trading_bot.cli backtest \
  --csv data/btc_usdt_15m.csv \
  --model models/btc.joblib \
  --trades-out reports/trades.csv \
  --equity-out reports/equity.csv
```

Run validation independently and optionally save its report:

```bash
python -m trading_bot.cli walk-forward \
  --csv data/btc_usdt_15m.csv \
  --report-out reports/walk-forward.json
```

## Live market mode

The default configuration connects to public Binance market data and simulates
orders using a persistent `$1,000` paper balance. It uses only completed candles,
so a signal from one candle is acted on at the beginning of the next polling cycle.

Process one candle as a smoke test:

```bash
python -m trading_bot.cli live --model models/btc.joblib --once
```

Then keep it running:

```bash
python -m trading_bot.cli live --model models/btc.joblib
```

The state is saved to `reports/live_state.json`, and audit events are appended to
`reports/trades.jsonl`. Use `--state` to give separate
strategies or symbols separate state files. The exchange, symbol, and timeframe
can be changed in `configs/default.yaml` or with these environment variables:

```bash
export EXCHANGE_ID=kraken
export TRADING_SYMBOL=BTC/USD
export TRADING_TIMEFRAME=15m
```

## Real orders

Use an exchange sandbox first by setting `live.sandbox: true`. Real/sandbox orders
require all three safeguards:

1. Set `live.dry_run: false` in the selected YAML config.
2. Export `EXCHANGE_API_KEY` and `EXCHANGE_API_SECRET` with trade-only permissions.
3. Pass `--execute` to the live command.

```bash
export EXCHANGE_API_KEY='...'
export EXCHANGE_API_SECRET='...'
python -m trading_bot.cli --config configs/default.yaml live \
  --model models/btc.joblib \
  --execute
```

Do not grant withdrawal permissions. The bot allocates at most
`live.max_position_fraction` of available equity, then reduces that amount when
the configured account risk and ATR-derived stop require it. It only sells the
position amount recorded in its state file. Stop the process with Ctrl+C; its
state is retained and exchange orders are reconciled on restart.

When the exchange reports attached protection support, the entry includes native
stop-loss and take-profit parameters. Set `live.require_native_protection: true`
to reject entries on exchanges without that feature. Client-side trailing and
break-even checks still require the bot to remain running. Gaps, slippage,
outages, or rejected orders can make realized losses exceed configured limits.

## Portfolio mode

Run separately trained models under shared portfolio exposure and correlation limits:

```bash
python -m trading_bot.cli portfolio-live \
  --strategy BTC/USDT=models/btc.joblib \
  --strategy ETH/USDT=models/eth.joblib
```

Use `--once` for a dry-run smoke test. Real portfolio orders require the same
configuration, API credentials, and `--execute` safeguards as single-symbol mode.

## Journal, alerts, and dashboard

Read recent activity from the terminal:

```bash
python -m trading_bot.cli journal
```

Telegram activates when both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set.
Email activates with `SMTP_HOST`, `ALERT_EMAIL_FROM`, and `ALERT_EMAIL_TO`; optional
SMTP settings are `SMTP_PORT`, `SMTP_USER`, and `SMTP_PASSWORD`. Alert delivery is
best-effort and never contains exchange API credentials.

Start the local dashboard:

```bash
python -m trading_bot.cli dashboard --port 8000
```

Open `http://127.0.0.1:8000`. The emergency halt blocks new entries but never
liquidates positions. Binding beyond localhost requires `DASHBOARD_TOKEN`; use a
TLS reverse proxy for any remote access. External uptime monitors can check
`/healthz`; it returns an error when strategy heartbeat state is missing or stale.

## Configuration

Strategy and execution defaults live in `configs/default.yaml`:

- `model.probability_threshold` controls long/flat/exit signals.
- `backtest.stop_loss_pct` is also enforced by the live runner on every poll.
- `live.max_position_fraction` caps each spot entry.
- `live.require_native_protection` fails closed when native stops are unavailable.
- `live.poll_interval_seconds` controls exchange polling.
- `live.candle_limit` must remain large enough for the 96-bar features.
- `risk.*` controls account risk, profit management, and circuit breakers.
- `portfolio.*` controls aggregate and correlated exposure.
- `validation.*` defines model deployment thresholds.

This project is educational software, not financial advice.
