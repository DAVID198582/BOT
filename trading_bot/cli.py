from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from trading_bot.backtest import run_backtest
from trading_bot.bot import PaperTradingBot
from trading_bot.config import load_config
from trading_bot.dashboard import DashboardData, serve_dashboard
from trading_bot.data import load_ohlcv_csv, split_train_test
from trading_bot.features import make_features, summarize_market
from trading_bot.live import LiveTradingRunner, StateStore, create_exchange, run_loop
from trading_bot.journal import TradeJournal
from trading_bot.models import (
    evaluate_model,
    load_model,
    predict_signal,
    save_model,
    train_model,
)
from trading_bot.portfolio import PortfolioStore
from trading_bot.validation import walk_forward_validate


def main() -> None:
    parser = argparse.ArgumentParser(description="AI crypto trading bot CLI")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Summarize historical market data")
    analyze_parser.add_argument("--csv", required=True)

    train_parser = subparsers.add_parser("train", help="Train an ML direction model")
    train_parser.add_argument("--csv", required=True)
    train_parser.add_argument("--model-out", required=True)
    train_parser.add_argument(
        "--allow-weak-model",
        action="store_true",
        help="Save a model that did not pass walk-forward thresholds",
    )

    backtest_parser = subparsers.add_parser("backtest", help="Backtest trained model signals")
    backtest_parser.add_argument("--csv", required=True)
    backtest_parser.add_argument("--model", required=True)
    backtest_parser.add_argument("--trades-out")
    backtest_parser.add_argument("--equity-out")

    paper_parser = subparsers.add_parser(
        "paper", help="Make a paper-trading decision from latest data"
    )
    paper_parser.add_argument("--csv", required=True)
    paper_parser.add_argument("--model", required=True)

    live_parser = subparsers.add_parser(
        "live", help="Poll completed exchange candles and trade (dry-run by default)"
    )
    live_parser.add_argument("--model", required=True)
    live_parser.add_argument(
        "--once", action="store_true", help="Process one completed candle and exit"
    )
    live_parser.add_argument(
        "--execute",
        action="store_true",
        help="Place real orders (also requires live.dry_run: false and API credentials)",
    )
    live_parser.add_argument("--state", help="Override the live state file path")
    live_parser.add_argument(
        "--allow-unapproved-model",
        action="store_true",
        help="Run a model without approved walk-forward metadata",
    )

    validation_parser = subparsers.add_parser(
        "walk-forward", help="Run expanding-window model validation"
    )
    validation_parser.add_argument("--csv", required=True)
    validation_parser.add_argument("--report-out")

    portfolio_parser = subparsers.add_parser(
        "portfolio-live", help="Run multiple SYMBOL=MODEL strategies"
    )
    portfolio_parser.add_argument(
        "--strategy", action="append", required=True, help="For example BTC/USD=models/btc.joblib"
    )
    portfolio_parser.add_argument("--once", action="store_true")
    portfolio_parser.add_argument("--execute", action="store_true")
    portfolio_parser.add_argument("--allow-unapproved-model", action="store_true")

    dashboard_parser = subparsers.add_parser(
        "dashboard", help="Serve the monitoring and emergency-halt dashboard"
    )
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8000)
    dashboard_parser.add_argument("--state-glob", default="reports/*state.json")

    journal_parser = subparsers.add_parser("journal", help="Show trade journal summary")
    journal_parser.add_argument("--limit", type=int, default=30)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "analyze":
        frame = load_ohlcv_csv(args.csv)
        print_json(summarize_market(frame))
        return

    if args.command == "train":
        frame = make_features(load_ohlcv_csv(args.csv))
        validation = walk_forward_validate(frame, config)
        if not validation["approved"] and not args.allow_weak_model:
            print_json({"saved": False, "validation": validation})
            raise SystemExit("model rejected by walk-forward validation")
        train_frame, test_frame = split_train_test(frame, config.data.test_size)
        model = train_model(train_frame, config.model)
        metrics = evaluate_model(model, test_frame)
        save_model(
            model,
            args.model_out,
            {
                "metrics": metrics,
                "feature_rows": len(frame),
                "config": args.config,
                "validation": validation,
            },
        )
        print_json({"saved": True, "metrics": metrics, "validation": validation})
        return

    if args.command == "backtest":
        model, metadata = load_model(args.model)
        frame = make_features(load_ohlcv_csv(args.csv))
        _, test_frame = split_train_test(frame, config.data.test_size)
        signals = predict_signal(model, test_frame, config.model.probability_threshold)
        result = run_backtest(test_frame, signals, config.backtest, config.risk)
        write_optional_csv(result.trades, args.trades_out)
        write_optional_csv(result.equity_curve, args.equity_out)
        print_json({"metrics": result.metrics, "model_metadata": metadata})
        return

    if args.command == "paper":
        model, _ = load_model(args.model)
        frame = load_ohlcv_csv(args.csv)
        decision = PaperTradingBot(model, config).decide(frame)
        print_json(decision.__dict__)
        return

    if args.command == "live":
        model, metadata = load_model(args.model)
        assert_approved_model(config, metadata, args.allow_unapproved_model)
        exchange = create_exchange(config, execute=args.execute)
        state_store = StateStore(
            args.state or config.live.state_path,
            config.data.symbol,
            not args.execute,
            config.backtest.initial_cash,
        )
        runner = LiveTradingRunner(
            model, config, exchange, execute=args.execute, state_store=state_store
        )
        try:
            run_loop(runner, print_json, once=args.once)
        except KeyboardInterrupt:
            print_json({"status": "stopped"})
        return

    if args.command == "walk-forward":
        frame = make_features(load_ohlcv_csv(args.csv))
        report = walk_forward_validate(frame, config)
        if args.report_out:
            output_path = Path(args.report_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
        print_json(report)
        return

    if args.command == "portfolio-live":
        run_portfolio(config, args)
        return

    if args.command == "dashboard":
        data = DashboardData(
            args.state_glob,
            config.portfolio.state_path,
            config.live.journal_path,
        )
        print_json({"status": "starting", "url": f"http://{args.host}:{args.port}"})
        serve_dashboard(data, args.host, args.port)
        return

    if args.command == "journal":
        journal = TradeJournal(config.live.journal_path)
        print_json(
            {"summary": journal.summary(), "recent": journal.records(args.limit)}
        )


def assert_approved_model(
    config, metadata: dict, allow_unapproved: bool
) -> None:
    if not config.live.require_approved_model or allow_unapproved:
        return
    validation = metadata.get("validation") or {}
    if not validation.get("approved"):
        raise ValueError(
            "Model lacks approved walk-forward validation; retrain it or pass "
            "--allow-unapproved-model explicitly"
        )


def run_portfolio(config, args) -> None:
    strategies: list[tuple[str, str]] = []
    for value in args.strategy:
        if "=" not in value:
            raise ValueError("--strategy must use SYMBOL=MODEL format")
        symbol, model_path = value.split("=", 1)
        strategies.append((symbol.strip(), model_path.strip()))
    exchange = create_exchange(config, execute=args.execute)
    shared_portfolio = PortfolioStore(
        config.portfolio.state_path, config.portfolio
    )
    runners: list[LiveTradingRunner] = []
    for symbol, model_path in strategies:
        model, metadata = load_model(model_path)
        assert_approved_model(config, metadata, args.allow_unapproved_model)
        safe_symbol = symbol.replace("/", "-").replace(":", "-")
        strategy_config = replace(
            config,
            data=replace(config.data, symbol=symbol),
            live=replace(
                config.live,
                state_path=f"reports/live-{safe_symbol}-state.json",
            ),
        )
        runners.append(
            LiveTradingRunner(
                model,
                strategy_config,
                exchange,
                execute=args.execute,
                portfolio=shared_portfolio,
            )
        )
    while True:
        for runner in runners:
            try:
                print_json(runner.run_once())
            except Exception as error:
                runner.record_error(error)
                if args.once:
                    raise
                print_json(
                    {
                        "status": "error",
                        "symbol": runner.config.data.symbol,
                        "error": str(error),
                    }
                )
        if args.once:
            return
        time.sleep(config.live.poll_interval_seconds)


def print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str), flush=True)


def write_optional_csv(frame, path: str | None) -> None:
    if path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
