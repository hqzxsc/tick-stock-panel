"""QMT / TDX 数据源字段与单位契约测试。

覆盖: TDX 日线/分钟 Volume 股→手、实时 Volume 总手原样、实时 Amount 万元→元、
实时股票池为全 A 股; QMT 复权因子索引日期解析、缺 dr 兜底、无昨收振幅。
不依赖真实 QMT/通达信客户端。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from app.plugins.qmt.provider import QMTProvider, _divid_factors_to_frame
from app.plugins.tdx import provider as tdx_provider
from app.plugins.tdx.provider import TDXProvider, _convert_kline_to_rows_fast

SYMBOL = "600000.SH"


def _tdx_frame(values: dict, index) -> dict:
    return {
        name: pd.DataFrame({SYMBOL: [value]}, index=index)
        for name, value in values.items()
    }


def test_tdx_daily_volume_shares_to_hands():
    data = _tdx_frame(
        {
            "Open": 10.0,
            "High": 10.5,
            "Low": 9.8,
            "Close": 10.2,
            "Volume": "2257325",
            "Amount": "29700.0",
        },
        pd.DatetimeIndex(["2026-09-07"]),
    )
    rows = _convert_kline_to_rows_fast(data, [SYMBOL])
    assert rows[0]["volume"] == 22573  # 2257325 股 → 22573 手
    assert rows[0]["amount"] == 297_000_000.0  # 万元 → 元

    data_odd = dict(data, Volume=pd.DataFrame({SYMBOL: ["99"]}, index=data["Close"].index))
    assert _convert_kline_to_rows_fast(data_odd, [SYMBOL])[0]["volume"] == 0


def test_tdx_minute_volume_shares_to_hands():
    data = _tdx_frame(
        {
            "Open": 10.0,
            "High": 10.1,
            "Low": 9.9,
            "Close": 10.0,
            "Volume": "1234567",
            "Amount": "1000.0",
        },
        pd.DatetimeIndex(["2026-09-07 09:31:00"]),
    )
    rows = TDXProvider._convert_minute_rows(data, [SYMBOL])
    assert rows[0]["volume"] == 12345
    assert rows[0]["amount"] == 10_000_000.0


class _FakeSnapshotTq:
    def __init__(self, snapshot: dict) -> None:
        self.snapshot = snapshot

    def get_market_snapshot(self, stock_code: str) -> dict:
        return dict(self.snapshot)


def test_tdx_realtime_units():
    snapshot = {
        "ErrorId": "0",
        "Now": "35.06",
        "LastClose": "34.0",
        "Max": "36.0",
        "Min": "33.0",
        "Open": "34.5",
        "Volume": "122881",  # 总手
        "Amount": "43068.48",  # 万元
    }
    rows = TDXProvider()._snapshot_rows(_FakeSnapshotTq(snapshot), [SYMBOL])
    row = rows[0]
    assert row["volume"] == 122881
    assert row["amount"] == pytest.approx(430_684_800.0)
    assert row["amplitude"] == pytest.approx((36.0 - 33.0) / 34.0)

    zero_snapshot = dict(snapshot, Volume="0", Amount="0")
    zero_row = TDXProvider()._snapshot_rows(_FakeSnapshotTq(zero_snapshot), [SYMBOL])[0]
    assert zero_row["volume"] == 0
    assert zero_row["amount"] == 0


def test_tdx_realtime_stock_list_uses_all_a_shares(monkeypatch):
    calls: list[str] = []

    class _FakeTq:
        def get_stock_list(self, market: str = "", list_type: int = 0):
            calls.append(market)
            return []

    monkeypatch.setattr(tdx_provider, "_get_tq", lambda: _FakeTq())
    TDXProvider().get_realtime()
    assert calls == ["5"]


def test_qmt_adj_factors_accepts_index_dates_and_dr():
    for index_value in ("2020-07-14", "20200714", 20200714):
        df = pd.DataFrame(
            {"interest": [0.0], "dr": [1.02]},
            index=pd.Index([index_value]),
        )
        out = _divid_factors_to_frame(df, SYMBOL)
        assert out.height == 1
        assert out["trade_date"][0] == dt.date(2020, 7, 14)
        assert out["ex_factor"][0] == 1.02
        assert out["symbol"][0] == SYMBOL


def test_qmt_adj_factors_without_dr_returns_empty():
    df = pd.DataFrame({"interest": [0.0]}, index=pd.Index(["2020-07-14"]))
    assert _divid_factors_to_frame(df, SYMBOL).is_empty()


def test_qmt_adj_factors_one_symbol_failure_keeps_others(monkeypatch):
    import sys
    import types

    class _FakeXtdata:
        def get_divid_factors(self, symbol, start_time, end_time):
            if symbol == "BAD.SH":
                raise RuntimeError("boom")
            return pd.DataFrame({"dr": [1.01]}, index=pd.Index(["2020-07-14"]))

    fake_module = types.ModuleType("xtquant")
    fake_module.xtdata = _FakeXtdata()
    monkeypatch.setitem(sys.modules, "xtquant", fake_module)

    out = QMTProvider().get_adj_factors(["600000.SH", "BAD.SH", "000001.SZ"], None, None)
    assert out.height == 2
    assert set(out["symbol"].to_list()) == {"600000.SH", "000001.SZ"}


def test_qmt_amplitude_none_when_prev_close_missing():
    class _FakeXtdata:
        def get_full_tick(self, symbols):
            return {
                SYMBOL: {
                    "lastPrice": 0.0,
                    "lastClose": 0.0,
                    "open": 0.0,
                    "high": 0.0,
                    "low": 0.0,
                    "volume": 0,
                    "amount": 0.0,
                    "time": 0,
                }
            }

    rows = QMTProvider()._tick_rows(_FakeXtdata(), [SYMBOL])
    assert rows[0]["amplitude"] is None
