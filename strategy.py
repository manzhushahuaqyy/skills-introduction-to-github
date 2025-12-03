"""Grid trading strategy implementation using Qlib's backtesting interfaces."""
from __future__ import annotations

from dataclasses import dataclass
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from qlib.backtest.account import Account
from qlib.backtest.decision import Order, OrderHelper, TradeDecisionWO
from qlib.backtest.exchange import Exchange
from qlib.backtest import backtest_loop
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.utils import CommonInfrastructure
from qlib.data import D
from qlib.strategy.base import BaseStrategy
from qlib.utils.time import Freq


@dataclass
class GridLevel:
    """Configuration for a single grid level."""

    anchor_price: float
    grid_size: float

    def should_buy(self, price: float) -> bool:
        return price <= self.anchor_price * (1 - self.grid_size)

    def should_sell(self, price: float) -> bool:
        return price >= self.anchor_price * (1 + self.grid_size)

    def update_anchor(self, price: float) -> None:
        self.anchor_price = price


class GridStrategy(BaseStrategy):
    """A grid trading strategy driven by daily close prices."""

    def __init__(
        self,
        close_signals: pd.DataFrame,
        grid_size: float = 0.02,
        cash_per_grid: float = 10000,
        min_trade_shares: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.close_signals = close_signals.sort_index()
        self.grid_size = grid_size
        self.cash_per_grid = cash_per_grid
        self.min_trade_shares = min_trade_shares
        self._levels: Dict[str, GridLevel] = {}

    def _get_level(self, instrument: str, price: float) -> GridLevel:
        if instrument not in self._levels:
            self._levels[instrument] = GridLevel(anchor_price=price, grid_size=self.grid_size)
        return self._levels[instrument]

    def _iter_current_prices(self, trade_time: pd.Timestamp) -> Iterable[tuple[str, float]]:
        if trade_time not in self.close_signals.index:
            return []
        row = self.close_signals.loc[trade_time]
        return row.dropna().items()

    def _position_amount(self, code: str) -> float:
        return float(self.trade_position.get_stock_amount(code))

    def _create_buy_order(self, code: str, price: float) -> Optional[Order]:
        target_amount = max(int(self.cash_per_grid / price), self.min_trade_shares)
        return OrderHelper.create(code=code, amount=target_amount, direction=Order.BUY)

    def _create_sell_order(self, code: str, price: float, current_amount: float) -> Optional[Order]:
        target_amount = min(max(int(self.cash_per_grid / price), self.min_trade_shares), int(current_amount))
        if target_amount <= 0:
            return None
        return OrderHelper.create(code=code, amount=target_amount, direction=Order.SELL)

    def _current_trade_time(self) -> pd.Timestamp:
        """Fetch the current trade time without requiring a look-ahead slot.

        Older Qlib builds advance the calendar index to the final entry before
        `generate_trade_decision` runs, which means ``get_step_time`` can raise
        ``IndexError`` when it tries to peek at ``index + 1``. This helper clamps
        the calendar index to the valid range and only calls ``get_step_time`` as
        a guarded last resort.
        """

        # Prefer the public ``get`` accessor when available; some implementations
        # safely clamp to the current calendar entry even at the final step.
        if hasattr(self.trade_calendar, "get"):
            try:
                return self.trade_calendar.get()
            except IndexError:
                # Fall back to raw fields below if a final-step lookup occurs.
                pass

        # Fall back to the raw calendar fields. Clamp the index to the valid range
        # to avoid out-of-bounds access at the last calendar entry.
        calendar = getattr(self.trade_calendar, "_calendar", None)
        if calendar is None:
            calendar = getattr(self.trade_calendar, "calendar", None)
        index = getattr(self.trade_calendar, "calendar_index", None)
        if calendar is not None and len(calendar) > 0:
            clamped_index = min(max(index or 0, 0), len(calendar) - 1)
            return calendar[clamped_index]

        # As a last resort, reuse ``get_step_time`` but guard against bounds.
        if hasattr(self.trade_calendar, "get_step_time"):
            try:
                trade_time, _ = self.trade_calendar.get_step_time()
                return trade_time
            except IndexError:
                if calendar is not None and len(calendar) > 0:
                    return calendar[-1]

        raise RuntimeError("Unable to resolve current trade time from calendar")

    def _ensure_calendar_padding(self) -> None:
        """Pad the calendar to avoid look-ahead index errors.

        ``TradeDecisionWO`` calls ``trade_calendar.get_step_time`` during
        initialization, which in some Qlib releases raises ``IndexError`` when
        the calendar index points to the final entry. Padding the calendar with
        a duplicate final timestamp ensures ``index + 1`` remains valid without
        introducing additional simulated time.
        """

        calendar = getattr(self.trade_calendar, "_calendar", None)
        if calendar is None:
            calendar = getattr(self.trade_calendar, "calendar", None)

        index = getattr(self.trade_calendar, "calendar_index", None)
        try:
            length = len(calendar) if calendar is not None else 0
        except TypeError:
            length = 0

        if calendar is None or index is None or length == 0:
            return

        if index < length - 1:
            return

        last_entry = calendar[-1]
        if isinstance(calendar, np.ndarray):
            padded = np.concatenate([calendar, calendar[-1:]])
        else:
            padded = list(calendar) + [last_entry]

        if hasattr(self.trade_calendar, "_calendar"):
            self.trade_calendar._calendar = padded
        if hasattr(self.trade_calendar, "calendar"):
            self.trade_calendar.calendar = padded

    def _patch_trade_calendar(self) -> None:
        """Monkey patch ``get_step_time`` to guard against look-ahead failures.

        Some Qlib builds call ``self._calendar[calendar_index + 1]`` inside
        ``get_step_time`` even when the calendar index already points to the
        final element. This wrapper clamps the index and falls back to the last
        calendar entry when an ``IndexError`` is raised.
        """

        tc = self.trade_calendar
        if tc is None or getattr(tc, "_grid_calendar_patched", False):
            return

        if not hasattr(tc, "get_step_time"):
            return

        original_get_step_time = tc.get_step_time

        def safe_get_step_time(self_tc):  # type: ignore[override]
            calendar = getattr(self_tc, "_calendar", getattr(self_tc, "calendar", []))
            try:
                cal_len = len(calendar)
            except Exception:
                cal_len = 0

            if cal_len:
                index = getattr(self_tc, "calendar_index", 0)
                clamped_index = min(max(index, 0), cal_len - 1)
                try:
                    self_tc.calendar_index = clamped_index
                except Exception:
                    pass

            try:
                return original_get_step_time()
            except IndexError:
                if cal_len:
                    last = calendar[-1]
                    return last, last
                raise

        tc.get_step_time = types.MethodType(safe_get_step_time, tc)
        tc._grid_calendar_patched = True

    def generate_trade_decision(self, execute_result: List[object] | None = None):  # type: ignore[override]
        self._ensure_calendar_padding()
        self._patch_trade_calendar()
        trade_time = self._current_trade_time()
        orders: List[Order] = []

        for code, price in self._iter_current_prices(trade_time):
            level = self._get_level(code, price)
            position_amount = self._position_amount(code)

            if level.should_buy(price):
                order = self._create_buy_order(code, price)
                if order:
                    orders.append(order)
                    level.update_anchor(price)
            elif level.should_sell(price) and position_amount > 0:
                order = self._create_sell_order(code, price, position_amount)
                if order:
                    orders.append(order)
                    level.update_anchor(price)

        # Avoid specifying ``trade_range`` because atomic executors reject
        # range limits. Calendar padding above prevents the look-ahead error
        # originally seen when ``trade_range`` was omitted.
        if not orders:
            return TradeDecisionWO(order_list=[], strategy=self)
        return TradeDecisionWO(order_list=orders, strategy=self)

    def backtest(
        self,
        executor: SimulatorExecutor,
        start_time: str,
        end_time: str,
    ):
        return backtest_loop(start_time=start_time, end_time=end_time, trade_strategy=self, trade_executor=executor)


def build_executor(
    start_time: str,
    end_time: str,
    codes: List[str],
    freq: str = "day",
    init_cash: float = 1_000_000.0,
) -> SimulatorExecutor:
    """Construct a Qlib executor wired to a simulated exchange and account."""

    exchange = Exchange(freq=freq, start_time=start_time, end_time=end_time, codes=codes, deal_price="$close")
    account = Account(init_cash=init_cash, benchmark_config={"benchmark": "SH000300"})
    # ``CommonInfrastructure`` accepts its components via keyword arguments in
    # older Qlib releases; use kwargs instead of attribute setters to remain
    # compatible with versions that do not expose a ``set`` helper.
    common = CommonInfrastructure(trade_account=account, trade_exchange=exchange)

    executor = SimulatorExecutor(
        time_per_step=freq,
        start_time=start_time,
        end_time=end_time,
        generate_portfolio_metrics=True,
        common_infra=common,
    )
    return executor


def _pick_column(columns, keywords: Iterable[str]):
    for col in columns:
        name = str(col).lower()
        if all(key.lower() in name for key in keywords):
            return col
    return None


def _calc_performance_metrics(report_df: pd.DataFrame) -> Dict[str, float]:
    """计算简化后的指标，仅保留用户需求。"""

    return_col = _pick_column(report_df.columns, ["return"]) or "return"
    bench_col = _pick_column(report_df.columns, ["bench"]) or _pick_column(report_df.columns, ["benchmark"])

    returns = report_df.get(return_col, pd.Series(dtype=float)).fillna(0.0)
    bench_returns = report_df.get(bench_col, pd.Series(dtype=float)).fillna(0.0)

    periods = len(returns)
    bench_curve = (1 + bench_returns).cumprod() if not bench_returns.empty else pd.Series(dtype=float)
    bench_return = bench_curve.iloc[-1] - 1 if not bench_curve.empty else 0.0

    return {
        "bench_return": bench_return,
        "periods": periods,
    }


def analysis_report(
    portfolio_dict,
    report_path: Path,
    freq: str = "day",
    *,
    start_time: str,
    end_time: str,
    init_cash: float,
    instruments: Iterable[str],
) -> Path:
    """生成包含中文标题与图表的回测报告 HTML。"""

    freq_key = f"{Freq.parse(freq)[0]}{Freq.parse(freq)[1]}"
    report_df, _ = portfolio_dict[freq_key]
    metrics = _calc_performance_metrics(report_df)

    return_col = _pick_column(report_df.columns, ["return"]) or "return"
    bench_col = _pick_column(report_df.columns, ["bench"]) or _pick_column(report_df.columns, ["benchmark"])
    returns = report_df.get(return_col, pd.Series(dtype=float)).fillna(0.0)
    bench_returns = report_df.get(bench_col, pd.Series(dtype=float)).fillna(0.0)

    strategy_curve = (1 + returns).cumprod()
    bench_curve = (1 + bench_returns).cumprod() if not bench_returns.empty else None

    def _fmt_date(value) -> str:
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)

    actual_start = _fmt_date(strategy_curve.index[0]) if not strategy_curve.empty else start_time
    actual_end = _fmt_date(strategy_curve.index[-1]) if not strategy_curve.empty else end_time

    # 收益曲线（含回撤填充，风格贴合示例）
    bench_curve = bench_curve if bench_curve is not None else pd.Series(dtype=float)
    bench_drawdown = bench_curve / bench_curve.cummax() - 1 if not bench_curve.empty else pd.Series(dtype=float)
    strat_drawdown = strategy_curve / strategy_curve.cummax() - 1 if not strategy_curve.empty else pd.Series(dtype=float)

    perf_fig = go.Figure()
    perf_fig.add_trace(
        go.Scatter(
            x=bench_curve.index,
            y=bench_curve.values,
            mode="lines",
            name="基准累计收益",
            line=dict(color="#5cb85c", width=2),
        )
    )
    perf_fig.add_trace(
        go.Scatter(
            x=strategy_curve.index,
            y=strategy_curve.values,
            mode="lines",
            name="策略累计收益",
            line=dict(color="#1976d2", width=2.4),
        )
    )
    perf_fig.add_trace(
        go.Scatter(
            x=strategy_curve.index,
            y=strat_drawdown.values,
            mode="lines",
            name="策略回撤",
            line=dict(color="rgba(255,99,132,0.8)", width=1.8),
            fill="tozeroy",
            fillcolor="rgba(255,99,132,0.18)",
        )
    )
    perf_fig.add_trace(
        go.Scatter(
            x=bench_drawdown.index,
            y=bench_drawdown.values,
            mode="lines",
            name="基准回撤",
            line=dict(color="rgba(255,193,7,0.9)", width=1.5, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(255,193,7,0.18)",
        )
    )
    perf_fig.add_trace(
        go.Scatter(
            x=strategy_curve.index,
            y=np.zeros_like(strategy_curve.values),
            mode="lines",
            name="零轴基线",
            line=dict(color="#9e9e9e", width=1, dash="dash"),
        )
    )
    perf_fig.update_layout(
        title=f"网格策略收益与回撤（{actual_start} 至 {actual_end}）",
        xaxis_title="日期",
        yaxis_title="累计收益/回撤",
        template="simple_white",
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(255,255,255,0.9)", bordercolor="#e0e0e0", borderwidth=1),
        margin=dict(l=60, r=30, t=80, b=60),
        height=520,
    )

    def fmt_pct(value: float) -> str:
        return f"{value:.2%}"

    freq_label = {"day": "日线", "1d": "日线"}.get(freq.lower(), freq)

    metric_cards = [
        ("沪深300累计收益率", fmt_pct(metrics["bench_return"])),
        ("回测周期", f"{actual_start} 至 {actual_end}"),
        ("回测天数", f"{metrics['periods']} 天"),
        ("交易频率", freq_label),
        ("初始资金", f"¥{init_cash:,.0f}"),
    ]

    style_block = """
    <style>
    body { font-family: 'Helvetica Neue', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f7f8fa; color: #2d3142; }
    .container { max-width: 1100px; margin: 24px auto 48px auto; padding: 0 16px; }
    .hero { text-align: center; margin-bottom: 16px; }
    .hero h1 { margin: 0; font-size: 28px; color: #1f2d3d; letter-spacing: 1px; }
    .subtitle { color: #606f7b; margin: 6px 0 0 0; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; margin: 22px 0 12px 0; }
    .card { background: white; border-radius: 12px; padding: 16px 14px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06); }
    .label { font-size: 14px; color: #607d8b; margin-bottom: 8px; }
    .value { font-size: 22px; font-weight: 700; color: #1b9aaa; }
    .chart { background: white; border-radius: 12px; padding: 10px 12px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06); margin-top: 16px; }
    </style>
    """

    metrics_html = "\n".join(
        f"<div class='card'><div class='label'>{label}</div><div class='value'>{value}</div></div>" for label, value in metric_cards
    )

    html_parts = [
        "<html><head>",
        style_block,
        "</head><body>",
        "<div class='container'>",
        "<div class='hero'>",
        "<h1>网格策略回测报告</h1>",
        f"<p class='subtitle'>回测区间：{actual_start} 至 {actual_end} ｜ 交易频率：{freq_label}</p>",
        "<p class='subtitle'>策略说明：依据每日收盘价设置上下网格，当价格触及网格下缘买入、上缘卖出，以分批交易平滑波动。</p>",
        "<p class='subtitle'>AI 参与说明：本报告约 30%-40% 内容由 AI 辅助生成，主要协助分析适配不同版本 Qlib 的指令差异并提供部分运行纠错建议，其余策略设计与参数由用户完成。</p>",
        "</div>",
        "<div class='metrics'>",
        metrics_html,
        "</div>",
        "<div class='chart'>",
        pio.to_html(perf_fig, full_html=False, include_plotlyjs="cdn"),
        "</div>",
        "</div>",
        "</body></html>",
    ]

    report_path.write_text("\n".join(html_parts), encoding="utf-8")
    return report_path


def load_close_signals(
    instruments: Iterable[str],
    features_root: Path,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> pd.DataFrame:
    """Load close price signals from binary feature files."""

    series_map: Dict[str, pd.Series] = {}
    for code in instruments:
        close_path = features_root / code / "close.day.bin"
        close_array = np.fromfile(close_path, dtype=np.float64)
        sliced = close_array[start_index:end_index]
        series_map[code] = pd.Series(sliced)
    return pd.DataFrame(series_map)


def load_hs300_instruments() -> List[str]:
    """Return the instrument list for the CSI 300 index."""

    instruments = D.instruments(market="csi300")

    # ``D.instruments`` can return different shapes depending on the Qlib version:
    # * older versions returned a DataFrame with an index of instrument codes
    # * some builds return a plain list of codes
    # * newer releases may return a mapping (e.g., {"csi300": [codes...]})
    if hasattr(instruments, "index"):
        inst_list = instruments.index.tolist()
    elif isinstance(instruments, dict):
        # Pull the first iterable of codes from the mapping
        for value in instruments.values():
            if isinstance(value, (list, tuple, set)):
                inst_list = list(value)
                break
        else:
            raise ValueError("Unexpected instrument mapping shape returned by D.instruments")
    elif isinstance(instruments, (list, tuple, set)):
        inst_list = list(instruments)
    else:
        raise TypeError(f"Unsupported instruments type: {type(instruments)!r}")

    return sorted(inst_list)
