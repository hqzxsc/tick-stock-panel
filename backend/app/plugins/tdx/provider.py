"""TDX (通达信 TdxQuant) 内置数据源 provider。

基于 tqcenter.py 的 tq 类实现,需本地运行通达信客户端。
API 文档: https://help.tdx.com.cn/quant/
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import polars as pl

from app.data_providers.base import AssetType
from app.data_providers.normalizer import normalize_adj_factors, normalize_daily
from app.plugins.tdx.bridge import get_process_dir_by_name
from app.tickflow.rate_limits import chunked

logger = logging.getLogger(__name__)

_DATASETS = ("adj_factor", "daily", "minute", "realtime", "instruments")
_BATCH = 100
_MINUTE_CANONICAL = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]

_tq_instance = None
_tq_initialized = False


@dataclass
class _TDXConfig:
    """轻量 config shim,便于 custom loader 与 router 识别。"""

    name: str = "tdx"
    display_name: str = "TDX (通达信 TdxQuant)"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def _get_tdx_root(exe_name: str) -> list[str]:
    """委托 bridge 做进程目录探测, 避免两处重复实现。"""
    return get_process_dir_by_name(exe_name)


def _finite_price(val) -> float:
    """把 TDX 返回的价格量规整为有限浮点数。

    None / NaN / Inf / 超出量级上限(|x| > max_finite) 的值统一转为 0.0
    (与代码库"停牌日 OHLC=0"约定一致, 进入下游后会被 filter_halt_days 剔除)。

    量级上限为 1e12: 真实 A 股日线 close ≤ ~3e3, 复权后 raw amount ≤ ~1e7,
    即便 amount 再乘 10000 也远低于 1e12; 而 TDX 偶发的损坏行会出现 ~1e27
    的异常大值, 用此上限拦截, 避免污染 _prev_raw_close / polars_limit_price 等计算。
    """
    max_finite = 1e12
    if val is None:
        return 0.0
    fv = float(val)
    return fv if math.isfinite(fv) and abs(fv) <= max_finite else 0.0


def _format_time(dt: datetime | None) -> str:
    """格式化时间为 tq 兼容的字符串 (YYYYMMDD 或 YYYYMMDDhhmmss)。"""
    if not dt:
        return ""
    return dt.strftime("%Y%m%d%H%M%S") if (dt.hour or dt.minute) else dt.strftime("%Y%m%d")


def _get_tq():
    """获取已初始化的 tq 实例,首次调用时自动初始化。"""
    global _tq_instance, _tq_initialized

    if _tq_instance is not None:
        return _tq_instance

    if _tq_initialized:
        return None

    try:
        tdx_root = _get_tdx_root("tdxw.exe")
        if len(tdx_root) == 0:
            logger.error("TDX: 未检测到通达信安装目录")
            _tq_initialized = True
            return None
        tdx_root = tdx_root[0]

        user_path = os.path.join(tdx_root, "PYPlugins", "user")
        if user_path not in sys.path:
            sys.path.insert(0, user_path)

        from tqcenter import tq

        # 用本 provider 文件自身作为 tq 策略占位 (tqcenter 基于 __file__ 上两级定位 DLL)
        anchor_file = os.path.abspath(__file__)
        logger.info("TDX: 开始初始化 tqcenter, tdx_root=%s, anchor=%s", tdx_root, anchor_file)
        tq.initialize(anchor_file)

        _tq_instance = tq
        _tq_initialized = True
        logger.info("TDX: tqcenter 初始化成功, tq id=%s", id(_tq_instance))
        # 验证 tpyth 是否就绪: 简单调用一次 snapshot 测试连接
        try:
            test_snap = tq.get_market_snapshot(stock_code="000001.SZ")
            if test_snap and str(test_snap.get("ErrorId", "0")) == "0":
                logger.info("TDX: 连接验证通过 (000001.SZ snapshot OK)")
            else:
                logger.warning("TDX: 连接验证异常 (000001.SZ snapshot=%s)", test_snap)
        except Exception as e:
            logger.warning("TDX: 连接验证失败: %s", e)
        return _tq_instance
    except Exception as e:
        logger.exception("TDX: tqcenter 初始化失败: %s", e)
        _tq_initialized = True
        return None


def _norm_date_val(val) -> str:
    """把日K日期索引归一为 'YYYY-MM-DD' (normalize_daily 的 cast(pl.Date) 口径)。

    tq 实测多为 DatetimeIndex; 兼容紧凑 'YYYYMMDD' 串/int —— 不归一时
    cast(pl.Date) 解析失败会静默丢行 (与 qmt 同款日期解析问题)。
    """
    try:
        return val.strftime("%Y-%m-%d")
    except AttributeError:
        s = str(val).strip()
        if "-" in s:
            return s[:10]
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if s[:8].isdigit() else s[:10]


def _norm_minute_val(val) -> str:
    """把分钟时间索引归一为 'YYYY-MM-DD HH:MM:SS' 北京墙钟字符串。

    兼容 DatetimeIndex 与紧凑 'YYYYMMDDHHMM(SS)' 串/int; 不归一时
    to_datetime 解析失败产出全空, 触发时区守卫回退 TickFlow。
    """
    try:
        return val.strftime("%Y-%m-%d %H:%M:%S")
    except AttributeError:
        s = str(val).strip()
        if "-" in s:
            return s
        digits = s.split(".", 1)[0]
        if digits.isdigit() and len(digits) == 12:  # YYYYMMDDHHMM
            digits += "00"
        if digits.isdigit() and len(digits) == 14:
            return (
                f"{digits[:4]}-{digits[4:6]}-{digits[6:8]} "
                f"{digits[8:10]}:{digits[10:12]}:{digits[12:14]}"
            )
        return s


def _convert_kline_to_rows_fast(data: dict, symbols: list[str]) -> list[dict]:
    """将 tq.get_market_data 返回的 dict 格式转为行列表(向量化版本)。"""
    if not data:
        return []

    close_df = data.get("Close")
    if close_df is None:
        return []

    import numpy as np

    rows: list[dict] = []
    for sym in symbols:
        if sym not in close_df.columns:
            continue
        try:
            dates = close_df.index
            n = len(dates)
            open_vals = (
                data["Open"][sym].values
                if "Open" in data and sym in data["Open"].columns
                else np.zeros(n)
            )
            high_vals = (
                data["High"][sym].values
                if "High" in data and sym in data["High"].columns
                else np.zeros(n)
            )
            low_vals = (
                data["Low"][sym].values
                if "Low" in data and sym in data["Low"].columns
                else np.zeros(n)
            )
            close_vals = data["Close"][sym].values
            vol_vals = (
                data["Volume"][sym].values
                if "Volume" in data and sym in data["Volume"].columns
                else np.zeros(n)
            )
            amt_vals = (
                data["Amount"][sym].values
                if "Amount" in data and sym in data["Amount"].columns
                else np.zeros(n)
            )

            for i in range(n):
                date_str = _norm_date_val(dates[i])
                rows.append(
                    {
                        "symbol": sym,
                        "date": date_str,
                        "open": _finite_price(open_vals[i]),
                        "high": _finite_price(high_vals[i]),
                        "low": _finite_price(low_vals[i]),
                        "close": _finite_price(close_vals[i]),
                        # 官方 Volume 单位:股, 内部契约:手 → 除以 100
                        "volume": math.floor(_finite_price(vol_vals[i]) / 100.0),
                        "amount": _finite_price(amt_vals[i]) * 10000.0,
                    }
                )
        except Exception as e:
            logger.warning("TDX: 转换 %s K线数据异常: %s", sym, e)
            continue

    return rows


class TDXProvider:
    """TDX (通达信 TdxQuant) 数据源 Provider。"""

    name = "tdx"
    builtin = True

    def __init__(self) -> None:
        self.config = _TDXConfig()

    def close(self) -> None:
        """清理资源。"""
        global _tq_instance, _tq_initialized
        if _tq_instance is not None:
            with contextlib.suppress(Exception):
                _tq_instance.close()
            _tq_instance = None
            _tq_initialized = False

    # ---- daily ----
    def get_daily(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()

        tq = _get_tq()
        if tq is None:
            return pl.DataFrame()

        logger.info("TDX daily 拉取开始(%d symbols)", len(symbols))
        frames: list[pl.DataFrame] = []
        chunks = chunked(symbols, _BATCH)
        start_str = _format_time(start_time)
        end_str = _format_time(end_time)

        for i, chunk in enumerate(chunks):
            try:
                data = tq.get_market_data(
                    field_list=["Open", "High", "Low", "Close", "Volume", "Amount"],
                    stock_list=chunk,
                    period="1d",
                    start_time=start_str,
                    end_time=end_str,
                    dividend_type="none",
                    fill_data=False,
                )
                rows = _convert_kline_to_rows_fast(data, chunk)
                if rows:
                    df = normalize_daily(rows, source=self.name)
                    if not df.is_empty():
                        frames.append(df)
            except Exception as e:
                logger.warning("TDX daily 拉取异常(%d symbols): %s", len(chunk), e)

            if on_chunk_done:
                on_chunk_done(i + 1, len(chunks))

        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    # ---- adj_factor ----
    def get_adj_factors(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()

        tq = _get_tq()
        if tq is None:
            return pl.DataFrame()

        frames: list[pl.DataFrame] = []
        chunks = chunked(symbols, _BATCH)
        start_str = _format_time(start_time)
        end_str = _format_time(end_time)

        for i, chunk in enumerate(chunks):
            try:
                data = tq.get_market_data(
                    field_list=["ForwardFactor"],
                    stock_list=chunk,
                    period="1d",
                    start_time=start_str,
                    end_time=end_str,
                    dividend_type="none",
                    fill_data=False,
                )
                df = data.get("ForwardFactor")
                if df is None or (hasattr(df, "is_empty") and df.is_empty()):
                    continue
                # pandas DataFrame: melt 转为长表
                df = df.melt(
                    value_vars=list(df.columns),
                    var_name="symbol",
                    value_name="ex_factor",
                    ignore_index=False,
                )
                df = df.reset_index()
                date_col = df.columns[0]
                df = df.rename(columns={date_col: "trade_date"})
                df = (
                    pl.from_pandas(df)
                    if hasattr(df, "to_pandas") or not hasattr(df, "with_columns")
                    else df
                )
                df = normalize_adj_factors(df, source=self.name)
                if df.is_empty():
                    continue
                # 先清洗非有限/非法因子(None 由 normalize 已 drop; NaN/Inf 在此处理),
                # 用前向填充补最近合法因子, 避免 NaN 污染下游 cum_prod / _prev_raw_close。
                df = df.with_columns(
                    pl.when(pl.col("ex_factor").is_nan() | pl.col("ex_factor").is_infinite())
                    .then(None)
                    .otherwise(pl.col("ex_factor"))
                    .alias("ex_factor"),
                ).sort(["symbol", "trade_date"])
                df = df.with_columns(
                    pl.col("ex_factor")
                    .forward_fill()
                    .over("symbol")
                    .fill_null(1.0)
                    .alias("ex_factor")
                )
                # TDX ForwardFactor 是"逐日绝对前复权因子"(水平值), 而下游 _apply_adj_factor
                # 按"除权日变动比"契约做 cum_prod 归一; 直接存水平值会让 0.7^k 下溢成 ~1e-25,
                # ratio 爆炸成 ~1e24, 复权价异常(如 000034.SZ 的 3e26)。
                # 这里把绝对因子转成相邻日比值: delta(d) = FF(d) / FF(d-1), 首个日期取 1.0。
                # delta 恒接近 1.0, cum_prod 不再下溢, 复权价保持真实数量级。
                df = df.with_columns(
                    (pl.col("ex_factor") / pl.col("ex_factor").shift(1).over("symbol"))
                    .fill_null(1.0)
                    .alias("ex_factor"),
                ).filter(pl.col("ex_factor").is_finite())
                if not df.is_empty():
                    frames.append(df)
            except Exception as e:
                logger.warning("TDX adj_factor 拉取异常(%d symbols): %s", len(chunk), e)

            if on_chunk_done:
                on_chunk_done(i + 1, len(chunks))

        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    # ---- minute ----
    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: AssetType = "stock",
        freq: str = "1m",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()

        tq = _get_tq()
        if tq is None:
            return pl.DataFrame()

        period = freq if freq.endswith("m") else f"{freq}m"
        logger.info("TDX minute 拉取开始(%d symbols, period=%s)", len(symbols), period)
        frames: list[pl.DataFrame] = []
        chunks = chunked(symbols, _BATCH)
        start_str = _format_time(start_time)
        end_str = _format_time(end_time)

        for i, chunk in enumerate(chunks):
            try:
                data = tq.get_market_data(
                    field_list=["Open", "High", "Low", "Close", "Volume", "Amount"],
                    stock_list=chunk,
                    period=period,
                    start_time=start_str,
                    end_time=end_str,
                    dividend_type="none",
                    fill_data=False,
                )
                rows = self._convert_minute_rows(data, chunk)
                if rows:
                    df = self._normalize_minute_rows(rows)
                    if not df.is_empty():
                        frames.append(df)
            except Exception as e:
                logger.warning("TDX minute 拉取异常(%d symbols): %s", len(chunk), e)

            if on_chunk_done:
                on_chunk_done(i + 1, len(chunks))

        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    @staticmethod
    def _convert_minute_rows(data: dict, symbols: list[str]) -> list[dict]:
        """将 tq.get_market_data 分钟数据转为行列表。"""
        if not data:
            return []

        import numpy as np

        close_df = data.get("Close")
        if close_df is None:
            return []

        rows: list[dict] = []
        for sym in symbols:
            if sym not in close_df.columns:
                continue
            try:
                dates = close_df.index
                n = len(dates)
                open_vals = (
                    data["Open"][sym].values
                    if "Open" in data and sym in data["Open"].columns
                    else np.zeros(n)
                )
                high_vals = (
                    data["High"][sym].values
                    if "High" in data and sym in data["High"].columns
                    else np.zeros(n)
                )
                low_vals = (
                    data["Low"][sym].values
                    if "Low" in data and sym in data["Low"].columns
                    else np.zeros(n)
                )
                close_vals = data["Close"][sym].values
                vol_vals = (
                    data["Volume"][sym].values
                    if "Volume" in data and sym in data["Volume"].columns
                    else np.zeros(n)
                )
                amt_vals = (
                    data["Amount"][sym].values
                    if "Amount" in data and sym in data["Amount"].columns
                    else np.zeros(n)
                )

                for i in range(n):
                    rows.append(
                        {
                            "symbol": sym,
                            "datetime": _norm_minute_val(dates[i]),
                            "open": float(open_vals[i]) if open_vals[i] is not None else 0.0,
                            "high": float(high_vals[i]) if high_vals[i] is not None else 0.0,
                            "low": float(low_vals[i]) if low_vals[i] is not None else 0.0,
                            "close": float(close_vals[i]) if close_vals[i] is not None else 0.0,
                            # 官方 Volume 单位:股, 内部契约:手 → 除以 100
                            "volume": math.floor(float(vol_vals[i]) / 100.0)
                            if vol_vals[i] is not None
                            else 0.0,
                            "amount": float(amt_vals[i]) * 10000.0
                            if amt_vals[i] is not None
                            else 0.0,
                        }
                    )
            except Exception as e:
                logger.warning("TDX: 转换 %s 分钟数据异常: %s", sym, e)
                continue
        return rows

    @staticmethod
    def _normalize_minute_rows(rows: list[dict]) -> pl.DataFrame:
        """将分钟行列表规范为 canonical schema。"""
        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)
        if df.is_empty():
            return df

        if "datetime" in df.columns:
            df = df.with_columns(
                pl.col("datetime").cast(pl.Utf8).str.to_datetime(strict=False).alias("datetime")
            )
            dt_type = df.schema["datetime"]
            if not isinstance(dt_type, pl.Datetime) or dt_type.time_unit != "us":
                df = df.with_columns(pl.col("datetime").cast(pl.Datetime("us"), strict=False))

        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        keep = [c for c in _MINUTE_CANONICAL if c in df.columns]
        return df.select(keep) if "datetime" in keep else pl.DataFrame()

    # ---- realtime ----
    def _snapshot_rows(self, tq, symbols: list[str]) -> list[dict]:
        """tq.get_market_snapshot → 契约行列表 (realtime 与 indices 共用)。

        单位契约: 快照 Volume 本身即"手", 原样入库; Amount 为万元→x10000 转元;
        change_pct 小数制按 change_amount/prev_close 推导。快照无服务端时间
        字段 → timestamp 按契约退本地毫秒 (quote_service 收盘定版确认依赖
        max(timestamp), 缺字段会导致收盘永不落盘)。
        """
        rows: list[dict] = []
        fail_count = 0
        for sym in symbols:
            try:
                snap = tq.get_market_snapshot(stock_code=sym)
                if not snap or str(snap.get("ErrorId", "0")) != "0":
                    fail_count += 1
                    continue

                last_price = float(snap.get("Now", 0) or 0)
                prev_close = float(snap.get("LastClose", 0) or 0)
                change_amount = None
                change_pct = None
                if last_price and prev_close:
                    change_amount = last_price - prev_close
                    change_pct = change_amount / prev_close if prev_close != 0 else 0.0
                high_val = float(snap.get("Max", 0) or 0)
                low_val = float(snap.get("Min", 0) or 0)
                amplitude = (
                    (high_val - low_val) / prev_close
                    if prev_close != 0 and high_val and low_val
                    else None
                )

                rows.append(
                    {
                        "symbol": sym,
                        "last_price": last_price,
                        "prev_close": prev_close,
                        "open": float(snap.get("Open", 0) or 0),
                        "high": high_val,
                        "low": low_val,
                        # 快照 Volume 单位为"总手", 内部契约同为"手" → 原样入库
                        "volume": float(snap.get("Volume", 0) or 0),
                        "amount": float(snap.get("Amount", 0) or 0) * 10000.0,
                        "change_amount": float(change_amount)
                        if change_amount is not None
                        else None,
                        "change_pct": float(change_pct) if change_pct is not None else None,
                        # 振幅按固定口径 (high-low)/prev_close 推导, 推导不了置 None 不伪造
                        "amplitude": float(amplitude) if amplitude is not None else None,
                        "timestamp": int(time.time() * 1000),
                    }
                )
            except Exception as e:
                fail_count += 1
                logger.debug("TDX snapshot %s 异常: %s", sym, e)
                continue
        if fail_count:
            logger.warning("TDX snapshot: %d/%d 只拉取失败", fail_count, len(symbols))
        return rows

    def get_realtime(self) -> list[dict]:
        """全市场 A 股实时快照 (纯股票, 指数走 get_realtime_indices 补拉协议)。

        失败软返回 [] (不阻断轮询线程)。ETF 实时未覆盖, 计数为 0。
        """
        tq = _get_tq()
        if tq is None:
            return []

        logger.info("TDX realtime 拉取开始(全市场股票快照)")
        try:
            # market='5'=全部 A 股(含北交所), 与 instruments 的股票池保持一致
            stock_list = tq.get_stock_list(market="5") or []
            if not stock_list:
                logger.warning(
                    "TDX realtime: get_stock_list(market='5') 返回空,请检查通达信是否运行并登录"
                )
                return []
            logger.info("TDX realtime: 待拉取 %d 只股票", len(stock_list))
            rows = self._snapshot_rows(tq, stock_list)
            logger.info("TDX realtime 完成: 成功 %d 只", len(rows))
            return rows
        except Exception as e:
            logger.warning("TDX realtime 拉取失败: %s", e)
            return []

    def get_realtime_indices(self, symbols: list[str]) -> list[dict] | None:
        """指数实时快照 (可选协议方法): 行字段与 get_realtime 一致。

        TDX 指数与股票同一 get_market_snapshot 通道, 按传入标准码逐只快照。
        失败返回 None (调用方保留上轮有效缓存), 成功但无数据返回 []。
        """
        if not symbols:
            return []
        tq = _get_tq()
        if tq is None:
            return None
        try:
            rows = self._snapshot_rows(tq, list(symbols))
            if not rows:
                # 全部失败按失败处理: 返回 None 让服务保留上轮指数缓存
                logger.warning("TDX realtime_indices: %d 只指数全部拉取失败", len(symbols))
                return None
            return rows
        except Exception as e:
            logger.warning("TDX realtime_indices 拉取失败: %s", e)
            return None

    # ---- instruments ----
    def get_instruments(self, asset_type: str = "stock") -> list[dict]:
        if asset_type != "stock":
            return []

        tq = _get_tq()
        if tq is None:
            return []

        try:
            stock_list = tq.get_stock_list(market="5", list_type=1) or []
            if not stock_list:
                logger.warning("TDX instruments: get_stock_list(market='5') 返回空")
                return []
            rows: list[dict] = []
            for item in stock_list:
                if isinstance(item, dict):
                    sym = item.get("Code", "")
                    name = item.get("Name", sym)
                else:
                    sym = str(item)
                    name = sym
                if not sym:
                    continue
                code = sym.split(".")[0] if "." in sym else sym
                exchange = sym.split(".")[1] if "." in sym else ""
                rows.append(
                    {
                        "symbol": sym,
                        "name": name,
                        "code": code,
                        "exchange": exchange,
                        "asset_type": "stock",
                        "source": self.name,
                    }
                )
            logger.info("TDX instruments: 共 %d 只", len(rows))
            return rows
        except Exception as e:
            logger.exception("TDX instruments 拉取失败: %s", e)
            return []

    # ---- test_dataset ----
    def test_dataset(self, dataset: str, symbols: list[str] | None = None) -> dict:
        symbols = symbols or ["600519.SH"]
        if dataset == "daily":
            df = self.get_daily(symbols, None, None)
            return _preview("daily", df)
        if dataset == "adj_factor":
            df = self.get_adj_factors(symbols, None, None)
            return _preview("adj_factor", df)
        if dataset == "minute":
            df = self.get_minute(symbols, None, None)
            return _preview("minute", df)
        if dataset == "realtime":
            rows = self.get_realtime()
            head = rows[:5]
            return {
                "provider": self.name,
                "dataset": "realtime",
                "rows": len(rows),
                "columns": list(head[0].keys()) if head else [],
                "preview": head,
            }
        if dataset == "instruments":
            rows = self.get_instruments("stock")
            head = rows[:5]
            return {
                "provider": self.name,
                "dataset": "instruments",
                "rows": len(rows),
                "columns": list(head[0].keys()) if head else [],
                "preview": head,
            }
        raise ValueError(f"TDX 不支持数据集: {dataset}")


def _preview(dataset: str, df: pl.DataFrame) -> dict:
    return {
        "provider": "tdx",
        "dataset": dataset,
        "rows": df.height,
        "columns": df.columns,
        "preview": df.head(5).to_dicts() if not df.is_empty() else [],
    }
