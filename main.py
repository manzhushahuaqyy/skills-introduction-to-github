"""Entry-point script that wires the grid strategy, data loading, and backtest."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.data import D

from strategy import (
    GridStrategy,
    analysis_report,
    build_executor,
    load_close_signals,
    load_hs300_instruments,
)


def load_feature_frame(stock_code: str, features_root: Path) -> pd.DataFrame:
    """Load feature columns for a single instrument using ``np.fromfile``.

    The expected directory layout is ``{features_root}/{stock_code}`` with files
    ``close.day.bin``, ``open.day.bin``, and ``volume.day.bin`` present.
    """

    fields = {
        "close": np.fromfile(features_root / stock_code / "close.day.bin", dtype=np.float64),
        "open": np.fromfile(features_root / stock_code / "open.day.bin", dtype=np.float64),
        "volume": np.fromfile(features_root / stock_code / "volume.day.bin", dtype=np.float64),
    }
    min_len = min(len(arr) for arr in fields.values())
    aligned = {name: values[:min_len] for name, values in fields.items()}
    return pd.DataFrame(aligned)


def prepare_signals(
    features_root: Path,
    instruments,
    start_time: str,
    end_time: str,
    freq: str = "day",
) -> pd.DataFrame:
    """Aggregate close-price signals for the supplied instrument list.

    If the expected local feature directory is unavailable, fall back to
    Qlib's data provider to pull ``$close`` features directly from the
    configured data source, enabling the backtest to run with any valid
    Qlib dataset (e.g., the default Yahoo bundle).
    """

    if features_root.exists():
        return load_close_signals(instruments=instruments, features_root=features_root)

    close_frames = []
    for code in instruments:
        features = D.features(code, fields=["$close"], start_time=start_time, end_time=end_time, freq=freq)
        # Align column name with instrument code for the combined DataFrame.
        close_frames.append(features.rename(columns={"$close": code}))
    return pd.concat(close_frames, axis=1)


def run_backtest() -> Path:
    """Execute the grid strategy backtest and create an HTML report."""

    qlib.init(provider_uri="D:/Python/Qlib/qlib_data", region=REG_CN)
    start_time = "2008-01-01"
    end_time = "2020-12-31"
    instruments = load_hs300_instruments()
    features_root = Path("D:/Python/Qlib/qlib_data/features")
    close_signals = prepare_signals(
        features_root=features_root,
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )

    init_cash = 1_000_000.0
    executor = build_executor(
        start_time=start_time,
        end_time=end_time,
        codes=instruments,
        freq="day",
        init_cash=init_cash,
    )
    strategy = GridStrategy(close_signals=close_signals, grid_size=0.03, cash_per_grid=5000)

    portfolio_dict, indicator_dict = strategy.backtest(
        executor=executor,
        start_time=start_time,
        end_time=end_time,
    )
    report_path = Path("网格策略回测报告.html")
    analysis_report(
        portfolio_dict,
        report_path=report_path,
        freq="day",
        start_time=start_time,
        end_time=end_time,
        init_cash=init_cash,
        instruments=instruments,
    )
    return report_path


if __name__ == "__main__":
    output = run_backtest()
    print(f"Backtest finished; report written to {output.resolve()}")
