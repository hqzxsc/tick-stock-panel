"""MINIQMT (迅投 xtquant) 内置数据源 provider。"""

from __future__ import annotations

import logging
import time as _time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import polars as pl

from app.data_providers.normalizer import normalize_adj_factors, normalize_daily
from app.tickflow.rate_limits import chunked

logger = logging.getLogger(__name__)

_DATASETS = ("adj_factor", "daily", "minute", "realtime", "instruments", "depth5")
_BATCH = 100
_MINUTE_CANONICAL = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount"]
_BJ_TZ = timezone(timedelta(hours=8))


@dataclass
class _QMTConfig:
    """轻量 config shim, 便于 custom loader 与 router 识别。"""

    name: str = "qmt"
    display_name: str = "QMT (迅投 MiniQMT)"
    datasets: dict = field(default_factory=lambda: dict.fromkeys(_DATASETS))
    path: None = None
    builtin: bool = True


def _format_time(dt: datetime | None) -> str:
    """格式化时间为 xtdata 兼容的字符串 (YYYYMMDD 或 YYYYMMDDhhmmss)。"""
    if not dt:
        return ""
    return dt.strftime("%Y%m%d%H%M%S") if (dt.hour or dt.minute) else dt.strftime("%Y%m%d")


def _daily_index_to_date(df_raw):
    """xtdata 日K的日期在未命名索引里 (值形如 '20260908'): reset 成 date 列。

    不处理这步时 normalize_daily 产出无 date 列, 落盘 partition_by("date") 会崩。
    'YYYYMMDD' 字符串 → ISO 'YYYY-MM-DD' (保证 normalize_daily 的 cast(pl.Date)
    可解析); datetime 索引与已含 '-' 的串直接保留。与 _minute_df 的 index 处理同思路。
    """
    df_raw = df_raw.reset_index()
    if "date" not in df_raw.columns and len(df_raw.columns):
        df_raw = df_raw.rename(columns={df_raw.columns[0]: "date"})
    if "date" not in df_raw.columns:
        return df_raw
    col = df_raw["date"]
    if str(col.dtype).startswith("datetime"):
        df_raw["date"] = col.dt.strftime("%Y-%m-%d")
        return df_raw
    s = col.astype(str)
    df_raw["date"] = s.where(
        s.str.contains("-"),
        s.str.slice(0, 4) + "-" + s.str.slice(4, 6) + "-" + s.str.slice(6, 8),
    )
    return df_raw


def _divid_factors_to_frame(df_fac, symbol: str) -> pl.DataFrame:
    """xtdata.get_divid_factors → 标准 adj_factor 表。

    官方返回的除权日在 DataFrame 索引里 (列为 interest/dr 等), 没有 time 列;
    必须先 reset_index 再做日期归一, 否则按 time 取列会 KeyError。
    """
    df = _daily_index_to_date(df_fac)
    df = df.rename(columns={"date": "trade_date", "dr": "ex_factor"})
    if "trade_date" not in df.columns or "ex_factor" not in df.columns:
        return pl.DataFrame()
    df["symbol"] = symbol
    return normalize_adj_factors(df, source="qmt")


def _parse_timetag(timetag) -> int | None:
    """把 xtdata 时间字段转为 int 毫秒 epoch。

    xtdata get_full_tick 的 timetag 实测为 'YYYYMMDD HH:MM:SS' 北京墙钟字符串,
    部分版本/场景也可能是 int/str 毫秒时间戳。返回 None 时由调用方回退。
    """
    if isinstance(timetag, (int, float)) and timetag:
        return int(timetag)
    if isinstance(timetag, str) and timetag:
        try:
            return int(timetag)
        except ValueError:
            pass
        try:
            dt = datetime.strptime(timetag, "%Y%m%d %H:%M:%S").replace(tzinfo=_BJ_TZ)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None
    return None


def _tick_ms(tick: dict) -> int | None:
    """从 get_full_tick 行提取毫秒时间戳: time 字段 (ms epoch) 优先, timetag 兜底。"""
    ts = tick.get("time")
    if isinstance(ts, (int, float)) and ts:
        return int(ts)
    return _parse_timetag(tick.get("timetag"))


class QMTProvider:
    """QMT (迅投 xtquant) 数据源 Provider。"""

    name = "qmt"
    builtin = True

    def __init__(self) -> None:
        self.config = _QMTConfig()
        # realtime 覆盖缺口告警只发一次 (轮询热路径 6s 一轮, 不能反复刷)
        self._coverage_warned = False

    def close(self) -> None:
        """清理资源。"""
        pass

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

        logger.info("QMT daily 拉取开始(%d symbols)", len(symbols))
        frames: list[pl.DataFrame] = []
        chunks = chunked(symbols, _BATCH)

        try:
            from xtquant import xtdata

            start_str = _format_time(start_time)
            end_str = _format_time(end_time)

            for i, chunk in enumerate(chunks):
                try:
                    raw_data = xtdata.get_market_data_ex(
                        field_list=["open", "high", "low", "close", "volume", "amount"],
                        stock_list=chunk,
                        period="1d",
                        start_time=start_str,
                        end_time=end_str,
                        dividend_type="none",
                    )
                    for sym, df_raw in raw_data.items():
                        if df_raw is None or len(df_raw) == 0:
                            continue

                        df_raw["symbol"] = sym
                        df = normalize_daily(_daily_index_to_date(df_raw), default_symbol=sym, source=self.name)
                        if not df.is_empty():
                            # 防御性兜底: amount 统一 Float64, 避免和仓库已有 parquet 的 Float64 列 concat 报错
                            if "amount" in df.columns and df.schema["amount"] != pl.Float64:
                                df = df.with_columns(
                                    pl.col("amount").cast(pl.Float64, strict=False)
                                )
                            frames.append(df)
                except Exception as e:
                    logger.warning("QMT daily 拉取异常(%d symbols): %s", len(chunk), e)

                if on_chunk_done:
                    on_chunk_done(i + 1, len(chunks))

        except Exception as e:
            logger.exception("QMT daily 拉取异常: %s", e)

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
        frames: list[pl.DataFrame] = []
        try:
            from xtquant import xtdata

            start_str = _format_time(start_time) if start_time else ""
            end_str = _format_time(end_time) if end_time else ""
            for i, sym in enumerate(symbols):
                try:
                    df_fac = xtdata.get_divid_factors(sym, start_str, end_str)
                    if df_fac is not None and not getattr(df_fac, "empty", True):
                        df = _divid_factors_to_frame(df_fac, sym)
                        if not df.is_empty():
                            frames.append(df)
                except Exception as e:
                    # 单只股票失败不影响其余股票, 避免整批退化为空
                    logger.warning("QMT adj_factor 拉取异常(%s): %s", sym, e)
                if on_chunk_done:
                    on_chunk_done(i + 1, len(symbols))
        except Exception as e:
            logger.exception("QMT adj_factors 拉取异常: %s", e)
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    @staticmethod
    def _minute_df(df_raw, symbol: str) -> pl.DataFrame:
        if df_raw is None or len(df_raw) == 0:
            return pl.DataFrame()
        if hasattr(df_raw, "reset_index"):
            df = pl.from_pandas(df_raw.reset_index())
        elif isinstance(df_raw, list):
            df = pl.DataFrame(df_raw)
        else:
            df = pl.DataFrame(df_raw)

        if df.is_empty():
            return pl.DataFrame()

        if "index" in df.columns and "datetime" not in df.columns and "time" not in df.columns:
            df = df.rename({"index": "datetime"})
        elif "time" in df.columns and "datetime" not in df.columns:
            df = df.rename({"time": "datetime"})

        if "datetime" in df.columns:
            dtype = df.schema["datetime"]
            if dtype in {pl.Int64, pl.Int32, pl.Float64}:
                # 毫秒时间戳 (UTC 口径) → +8h 北京墙钟
                df = df.with_columns(
                    (
                        pl.from_epoch(pl.col("datetime").cast(pl.Int64), time_unit="ms")
                        + pl.duration(hours=8)
                    ).alias("datetime")
                )
            elif dtype == pl.Utf8:
                # 兼容 '2026-09-07 09:30:00' 与 xtdata '20260907093000' (北京墙钟)
                df = df.with_columns(
                    pl.when(pl.col("datetime").str.contains("-", literal=True))
                    .then(
                        pl.col("datetime").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
                    )
                    .otherwise(
                        pl.col("datetime").str.to_datetime(format="%Y%m%d%H%M%S", strict=False)
                    )
                    .alias("datetime")
                )
            # pl.Datetime 输入视为北京墙钟 naive, 直接保留

        df = df.with_columns(pl.lit(symbol).alias("symbol"))
        for col in ("open", "high", "low", "close", "volume", "amount"):
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        keep = [c for c in _MINUTE_CANONICAL if c in df.columns]
        return df.select(keep) if "datetime" in keep else pl.DataFrame()

    # ---- minute ----
    def get_minute(
        self,
        symbols: list[str],
        start_time: datetime | None,
        end_time: datetime | None,
        asset_type: str = "stock",
        freq: str = "1m",
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> pl.DataFrame:
        if not symbols:
            return pl.DataFrame()

        logger.info("QMT minute 拉取开始(%d symbols, freq=%s)", len(symbols), freq)
        frames: list[pl.DataFrame] = []
        chunks = chunked(symbols, _BATCH)

        try:
            from xtquant import xtdata

            start_str = _format_time(start_time)
            end_str = _format_time(end_time)

            for i, chunk in enumerate(chunks):
                try:
                    # xtdata 历史K线需先下载到本地缓存, get_market_data_ex 才读得到
                    self._download_history(xtdata, chunk, freq, start_str, end_str)
                    raw_data = xtdata.get_market_data_ex(
                        field_list=["open", "high", "low", "close", "volume", "amount"],
                        stock_list=chunk,
                        period=freq,
                        start_time=start_str,
                        end_time=end_str,
                        dividend_type="none",
                    )
                    for sym, df_raw in raw_data.items():
                        df = self._minute_df(df_raw, sym)
                        if not df.is_empty():
                            frames.append(df)
                except Exception as e:
                    logger.warning("QMT minute 拉取异常(%d symbols): %s", len(chunk), e)

                if on_chunk_done:
                    on_chunk_done(i + 1, len(chunks))

        except Exception as e:
            logger.exception("QMT minute 拉取异常: %s", e)

        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    @staticmethod
    def _download_history(
        xtdata, symbols: list[str], period: str, start_str: str, end_str: str
    ) -> None:
        """下载历史K线到 xtdata 本地缓存。批量接口优先, 失败退化为逐标的下载。"""
        dl2 = getattr(xtdata, "download_history_data2", None)
        if callable(dl2):
            try:
                dl2(symbols, period, start_time=start_str, end_time=end_str)
                return
            except Exception as e:
                logger.warning("QMT download_history_data2 失败, 退化为逐标的下载: %s", e)
        for sym in symbols:
            try:
                xtdata.download_history_data(sym, period, start_time=start_str, end_time=end_str)
            except Exception as e:
                logger.warning("QMT download_history_data %s 失败: %s", sym, e)

    # ---- realtime ----
    def get_realtime(self) -> list[dict]:
        """全市场 A 股实时快照 (纯股票, 指数走 get_realtime_indices 补拉协议)。

        失败软返回 [] (不阻断轮询线程)。北交所标的快照取决于行情源,
        覆盖缺口只告警一次, 不在轮询热路径反复刷日志。
        """
        logger.info("QMT realtime 拉取开始(全市场股票快照)")
        try:
            from xtquant import xtdata

            stocks = xtdata.get_stock_list_in_sector("沪深京A股") or []
            if not stocks:
                logger.warning("QMT realtime 沪深京A股板块为空, 本轮跳过")
                return []
            rows = self._tick_rows(xtdata, stocks)
            # get_full_tick 逐标的返回, 无数据标的(如实测 .BJ)被静默丢弃 ——
            # 首次告警说明覆盖缺口, 后续轮次静默 (CONTRIBUTING: 不静默吞空数据)
            if len(rows) < len(stocks) and not self._coverage_warned:
                self._coverage_warned = True
                logger.warning(
                    "QMT realtime 快照覆盖不全: 请求 %d 只, 实得 %d 只 "
                    "(北交所等标的行情源可能未提供, 将持续缺失)",
                    len(stocks), len(rows),
                )
            return rows
        except Exception as e:
            logger.warning("QMT realtime 拉取失败: %s", e)
            return []

    def get_realtime_indices(self, symbols: list[str]) -> list[dict] | None:
        """指数实时快照 (可选协议方法): 行字段与 get_realtime 一致。

        QMT 指数与股票同一 get_full_tick 通道, 逐标的返回无连坐, 无效代码
        自然被丢弃。失败返回 None (调用方保留上轮有效缓存), 成功无数据返回 []。
        """
        if not symbols:
            return []
        try:
            from xtquant import xtdata

            return self._tick_rows(xtdata, list(symbols))
        except Exception as e:
            logger.warning("QMT realtime_indices 拉取失败: %s", e)
            return None

    def _tick_rows(self, xtdata, symbols: list[str]) -> list[dict]:
        """get_full_tick → 契约行列表 (realtime 与 indices 共用)。

        单位契约: volume 手 / amount 元 / change_pct 小数制 / timestamp 毫秒。
        """
        ticks = xtdata.get_full_tick(symbols)
        rows: list[dict] = []
        if not isinstance(ticks, dict):
            return rows
        for sym, tick in ticks.items():
            if not tick:
                continue
            last_price = tick.get("lastPrice", 0.0)
            prev_close = tick.get("lastClose", 0.0)
            change_amount = None
            change_pct = None
            if last_price and prev_close:
                change_amount = last_price - prev_close
                change_pct = change_amount / prev_close if prev_close != 0 else 0.0

            amplitude = (
                (tick.get("high", 0.0) - tick.get("low", 0.0)) / prev_close
                if prev_close != 0
                else None
            )

            rows.append(
                {
                    "symbol": sym,
                    "last_price": float(last_price),
                    "prev_close": float(prev_close),
                    "open": float(tick.get("open", 0.0)),
                    "high": float(tick.get("high", 0.0)),
                    "low": float(tick.get("low", 0.0)),
                    "volume": float(tick.get("volume", 0)),
                    "amount": float(tick.get("amount", 0.0)),
                    "change_amount": float(change_amount)
                    if change_amount is not None
                    else None,
                    "change_pct": float(change_pct) if change_pct is not None else None,
                    "amplitude": float(amplitude) if amplitude is not None else None,
                    # 服务端行情时间戳(ms), 缺失退本地时间 — quote_service
                    # final 定版确认依赖 max(timestamp), 缺字段会导致收盘永不落盘
                    "timestamp": _tick_ms(tick) or int(_time.time() * 1000),
                }
            )
        return rows

    # ---- depth5 ----
    def get_depth_batch(self, symbols: list[str]) -> dict[str, dict]:
        """五档盘口批量拉取 — 通过 xtdata.get_full_tick 获取五档量价。

        返回 {symbol: {bid_prices, ask_prices, bid_volumes, ask_volumes, timestamp}}
        其中 bid/ask_volumes 单位为"手"(CONTRIBUTING §3.1), 前端计算封单额时乘 100。
        depth_service 消费 ask_volumes/bid_volumes/timestamp 三个字段。
        """
        if not symbols:
            return {}
        try:
            from xtquant import xtdata

            ticks = xtdata.get_full_tick(symbols)
            if not isinstance(ticks, dict):
                return {}
            result: dict[str, dict] = {}
            for sym, tick in ticks.items():
                if not tick:
                    continue
                # get_full_tick 五档字段: bidPrice/bidVol/askPrice/askVol 均为长度 5 的列表;
                # time 为毫秒时间戳(int), timetag 为 'YYYYMMDD HH:MM:SS' 字符串。
                # 成交量单位为"手"。
                result[sym] = {
                    "bid_prices": list(tick.get("bidPrice") or []),
                    "ask_prices": list(tick.get("askPrice") or []),
                    "bid_volumes": list(tick.get("bidVol") or []),
                    "ask_volumes": list(tick.get("askVol") or []),
                    "timestamp": _tick_ms(tick),
                }
            return result
        except Exception as e:
            logger.warning("QMT depth5 拉取失败: %s", e)
            return {}

    # ---- instruments ----
    def get_instruments(self, asset_type: str = "stock") -> list[dict]:
        if asset_type != "stock":
            return []

        try:
            from xtquant import xtdata

            stocks = xtdata.get_stock_list_in_sector("沪深京A股")
            rows: list[dict] = []
            for sym in stocks:
                detail = xtdata.get_instrument_detail(sym) or {}
                code, _ = sym.split(".") if "." in sym else (sym, "")
                rows.append(
                    {
                        "symbol": sym,
                        "name": detail.get("InstrumentName") or sym,
                        "code": code,
                        "exchange": detail.get("ExchangeID"),
                        "asset_type": "stock",
                        "source": self.name,
                    }
                )
            return rows
        except Exception as e:
            logger.exception("QMT instruments 拉取失败: %s", e)
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
        if dataset == "depth5":
            data = self.get_depth_batch(symbols)
            head = [{k: v for k, v in d.items()} for d in list(data.values())[:5]]
            return {
                "provider": self.name,
                "dataset": "depth5",
                "rows": len(data),
                "columns": list(head[0].keys()) if head else [],
                "preview": head,
            }
        raise ValueError(f"QMT 不支持数据集: {dataset}")


def _preview(dataset: str, df: pl.DataFrame) -> dict:
    return {
        "provider": "qmt",
        "dataset": dataset,
        "rows": df.height,
        "columns": df.columns,
        "preview": df.head(5).to_dicts() if not df.is_empty() else [],
    }
