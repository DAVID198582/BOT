from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_bot.backtest import run_backtest
from trading_bot.bot import PaperTradingBot
from trading_bot.config import load_config
from trading_bot.data import load_ohlcv_csv, split_train_test
from trading_bot.features import make_features, summarize_market
from trading_bot.models import evaluate_model, load_model, predict_signal, save_model, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="AI crypto trading bot CLI")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Summarize historical market data")
    analyze_parser.add_argument("--csv", required=True)

    train_parser = subparsers.add_parser("train", help="Train an ML direction model")
    train_parser.add_argument("--csv", required=True)
    train_parser.add_argument("--model-out", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Backtest trained model signals")
    backtest_parser.add_argument("--csv", required=True)
    backtest_parser.add_argument("--model", required=True)
    backtest_parser.add_argument("--trades-out")
    backtest_parser.add_argument("--equity-out")

    paper_parser = subparsers.add_parser("paper", help="Make a paper-trading decision from latest data")
    paper_parser.add_argument("--csv", required=True)
    paper_parser.add_argument("--model", required=True)

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "analyze":
        frame = load_ohlcv_csv(args.csv)
        print_json(summarize_market(frame))
        return

    if args.command == "train":
        frame = make_features(load_ohlcv_csv(args.csv))
        train_frame, test_frame = split_train_test(frame, config.data.test_size)
        model = train_model(train_frame, config.model)
        metrics = evaluate_model(model, test_frame)
        save_model(
            model,
            args.model_out,
            {"metrics": metrics, "feature_rows": len(frame), "config": args.config},
        )
        print_json(metrics)
        return

    if args.command == "backtest":
        model, metadata = load_model(args.model)
        frame = make_features(load_ohlcv_csv(args.csv))
        _, test_frame = split_train_test(frame, config.data.test_size)
        signals = predict_signal(model, test_frame, config.model.probability_threshold)
        result = run_backtest(test_frame, signals, config.backtest)
        write_optional_csv(result.trades, args.trades_out)
        write_optional_csv(result.equity_curve, args.equity_out)
        print_json({"metrics": result.metrics, "model_metadata": metadata})
        return

    if args.command == "paper":
        model, _ = load_model(args.model)
        frame = load_ohlcv_csv(args.csv)
        decision = PaperTradingBot(model, config).decide(frame)
        print_json(decision.__dict__)


def print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def write_optional_csv(frame, path: str | None) -> None:
    if path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

