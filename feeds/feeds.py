"""
BMR — Data Feeds
=================
Ingestors for all three scale layers:

  PARTICIPANT  — OHLCV, sentiment, social, retail flow, options skew
  INSTITUTIONAL — COT commercial, dark pool proxies, credit spreads, fund flow
  SOVEREIGN    — Central bank stance, macro indicators, geopolitical stress

All feeds return a normalised RawSignal dataclass.
Data sourced from free/public APIs where possible; commercial feed
adapters marked clearly.

Evidence posture inherited from KINDFIELD:
  [ESTABLISHED]  — well-supported, reliable data
  [TESTABLE]     — directionally valid, requires calibration
  [SPECULATIVE]  — exploratory, must be clearly marked
"""

from __future__ import annotations
import os
import time
import logging
import requests
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ─── Core data types ──────────────────────────────────────────────────────────

@dataclass
class OHLCV:
    """Single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str


@dataclass
class RawSignal:
    """
    Normalised directional signal from any feed.
    value: -1.0 (max bearish/entropic) → +1.0 (max bullish/syntropic)
    """
    scale: str            # PARTICIPANT | INSTITUTIONAL | SOVEREIGN
    source: str           # feed name
    symbol: str
    value: float          # -1.0 → +1.0
    confidence: float     # 0.0 → 1.0 (data quality / recency)
    evidence_level: str   # ESTABLISHED | TESTABLE | SPECULATIVE
    timestamp: datetime
    raw: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# ─── PARTICIPANT LAYER ────────────────────────────────────────────────────────

class OHLCVFeed:
    """
    OHLCV data from Yahoo Finance (free) or Alpaca/Polygon (commercial).
    [ESTABLISHED] — price/volume is the ground truth of market behaviour.
    """

    def __init__(self, source: str = "yahoo"):
        self.source = source

    def fetch(self, symbol: str, timeframe: str = "1d",
              periods: int = 200) -> List[OHLCV]:
        """Fetch OHLCV bars. Returns list oldest→newest."""
        if self.source == "yahoo":
            return self._fetch_yahoo(symbol, timeframe, periods)
        raise ValueError(f"Unknown source: {self.source}")

    def _fetch_yahoo(self, symbol: str, timeframe: str,
                     periods: int) -> List[OHLCV]:
        """Yahoo Finance via yfinance library."""
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError("pip install yfinance")

        tf_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h",
            "4h": "4h", "1d": "1d", "1w": "1wk", "1mo": "1mo"
        }
        yf_tf = tf_map.get(timeframe, "1d")
        ticker = yf.Ticker(symbol)

        # Period string
        if timeframe in ("1m", "5m", "15m"):
            period = "7d"
        elif timeframe in ("1h", "4h"):
            period = "60d"
        else:
            period = "2y"

        df = ticker.history(period=period, interval=yf_tf)
        bars = []
        for ts, row in df.iterrows():
            bars.append(OHLCV(
                timestamp=ts.to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
                symbol=symbol,
                timeframe=timeframe,
            ))
        return bars[-periods:]


class MomentumSignal:
    """
    Participant momentum signal from price structure.
    Computes directional bias from RSI, MACD, rate-of-change.
    [ESTABLISHED] — momentum as directional indicator.
    """

    def compute(self, bars: List[OHLCV], symbol: str) -> RawSignal:
        if len(bars) < 26:
            return RawSignal(
                scale="PARTICIPANT", source="momentum",
                symbol=symbol, value=0.0, confidence=0.1,
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(),
                notes="Insufficient data"
            )

        closes = np.array([b.close for b in bars])

        # RSI-14
        rsi = self._rsi(closes, 14)

        # MACD histogram normalised
        macd_hist = self._macd_hist(closes)

        # Rate of change 10-period
        roc = (closes[-1] - closes[-11]) / (closes[-11] + 1e-10)
        roc_norm = float(np.clip(roc * 10, -1, 1))

        # Combine: RSI → -1..+1, MACD hist sign + magnitude
        rsi_norm = float((rsi - 50) / 50)  # 0→-1, 50→0, 100→+1
        macd_norm = float(np.clip(macd_hist / (np.std(closes) + 1e-10) * 5, -1, 1))

        value = float(np.clip(rsi_norm * 0.4 + macd_norm * 0.35 + roc_norm * 0.25, -1, 1))

        return RawSignal(
            scale="PARTICIPANT", source="momentum",
            symbol=symbol, value=value, confidence=0.75,
            evidence_level="ESTABLISHED",
            timestamp=bars[-1].timestamp,
            raw={"rsi": rsi, "roc": roc, "macd_hist": macd_hist},
        )

    def _rsi(self, closes: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _macd_hist(self, closes: np.ndarray) -> float:
        def ema(arr, n):
            k = 2 / (n + 1)
            e = arr[0]
            for v in arr[1:]:
                e = v * k + e * (1 - k)
            return e
        fast = ema(closes, 12)
        slow = ema(closes, 26)
        macd_line = fast - slow
        signal = ema(np.array([fast - slow]), 9)
        return float(macd_line - signal)


class VolumePressureSignal:
    """
    Volume-weighted buying/selling pressure.
    Positive: volume concentrated in upticks (accumulation).
    Negative: volume concentrated in downticks (distribution).
    [ESTABLISHED] — volume/price relationship.
    """

    def compute(self, bars: List[OHLCV], symbol: str) -> RawSignal:
        if len(bars) < 20:
            return RawSignal(
                scale="PARTICIPANT", source="volume_pressure",
                symbol=symbol, value=0.0, confidence=0.1,
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(),
            )

        recent = bars[-20:]
        obv_delta = 0.0
        for i in range(1, len(recent)):
            if recent[i].close > recent[i-1].close:
                obv_delta += recent[i].volume
            elif recent[i].close < recent[i-1].close:
                obv_delta -= recent[i].volume

        # Normalise against total volume
        total_vol = sum(b.volume for b in recent) + 1e-10
        value = float(np.clip(obv_delta / total_vol, -1, 1))

        return RawSignal(
            scale="PARTICIPANT", source="volume_pressure",
            symbol=symbol, value=value, confidence=0.70,
            evidence_level="ESTABLISHED",
            timestamp=recent[-1].timestamp,
            raw={"obv_delta": obv_delta, "total_vol": total_vol},
        )


class OptionsSkewSignal:
    """
    Put/call skew as participant fear/greed reading.
    Requires options data — uses Yahoo Finance for equity options,
    falls back to VIX/VVIX ratio proxy for index.
    [TESTABLE] — skew as directional predictor.
    """

    def compute(self, symbol: str) -> RawSignal:
        # VIX proxy for equity indices
        if symbol in ("SPY", "QQQ", "^GSPC", "^NDX"):
            return self._vix_proxy(symbol)
        return self._options_skew(symbol)

    def _vix_proxy(self, symbol: str) -> RawSignal:
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
            # VIX: low (<15) = complacency/bullish, high (>30) = fear/bearish
            value = float(np.clip(-(vix - 20) / 15, -1, 1))
            return RawSignal(
                scale="PARTICIPANT", source="options_skew",
                symbol=symbol, value=value, confidence=0.65,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                raw={"vix": vix},
                notes="VIX proxy — inverse: high VIX = bearish participant sentiment"
            )
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")
            return RawSignal(
                scale="PARTICIPANT", source="options_skew",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            )

    def _options_skew(self, symbol: str) -> RawSignal:
        """
        Simple put/call ratio from Yahoo options chain.
        Positive (call dominant) → bullish participant bias.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            if not expirations:
                raise ValueError("No options data")

            # Use nearest expiry
            chain = ticker.option_chain(expirations[0])
            call_vol = chain.calls["volume"].sum()
            put_vol = chain.puts["volume"].sum()
            total = call_vol + put_vol + 1e-10
            pc_ratio = put_vol / (call_vol + 1e-10)

            # PC ratio: <0.7 bullish, >1.2 bearish
            value = float(np.clip(-(pc_ratio - 0.9) / 0.5, -1, 1))
            return RawSignal(
                scale="PARTICIPANT", source="options_skew",
                symbol=symbol, value=value, confidence=0.60,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                raw={"pc_ratio": pc_ratio, "call_vol": call_vol, "put_vol": put_vol},
            )
        except Exception as e:
            logger.warning(f"Options skew failed for {symbol}: {e}")
            return RawSignal(
                scale="PARTICIPANT", source="options_skew",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            )


# ─── INSTITUTIONAL LAYER ─────────────────────────────────────────────────────

class COTSignal:
    """
    CFTC Commitments of Traders — commercial vs non-commercial positioning.
    Commercial hedgers are the smart money in commodities/futures.
    Non-commercial (speculators) provide contrarian signals at extremes.
    [ESTABLISHED] — COT as institutional positioning indicator.

    Data: CFTC public API (free, weekly, Tuesdays)
    """

    CFTC_URL = "https://www.cftc.gov/dea/newcot/deahistfo.zip"

    # CFTC market codes for common instruments
    MARKET_CODES = {
        "ES":  "13874A",  # E-mini S&P 500
        "NQ":  "209742",  # E-mini NASDAQ-100
        "GC":  "088691",  # Gold
        "CL":  "067651",  # Crude Oil WTI
        "EUR": "099741",  # Euro FX
        "GBP": "096742",  # British Pound
        "JPY": "097741",  # Japanese Yen
        "BTC": "133741",  # Bitcoin (CME)
    }

    def compute(self, symbol: str) -> RawSignal:
        market_code = self.MARKET_CODES.get(symbol.upper())
        if not market_code:
            return RawSignal(
                scale="INSTITUTIONAL", source="cot",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(),
                notes=f"No COT mapping for {symbol}"
            )
        try:
            return self._fetch_cot(symbol, market_code)
        except Exception as e:
            logger.warning(f"COT fetch failed for {symbol}: {e}")
            return RawSignal(
                scale="INSTITUTIONAL", source="cot",
                symbol=symbol, value=0.0, confidence=0.2,
                evidence_level="ESTABLISHED", timestamp=datetime.utcnow(),
            )

    def _fetch_cot(self, symbol: str, market_code: str) -> RawSignal:
        """
        Parse CFTC COT data.
        Net commercial position as % of open interest → institutional bias signal.
        """
        import urllib.request
        import zipfile
        import io
        import csv

        # Cache locally (COT is weekly)
        cache_path = f"/tmp/cot_cache_{market_code}.csv"
        cache_age = 0
        if os.path.exists(cache_path):
            cache_age = time.time() - os.path.getmtime(cache_path)

        if cache_age > 86400 * 3:  # refresh every 3 days
            try:
                with urllib.request.urlopen(self.CFTC_URL, timeout=15) as resp:
                    zdata = resp.read()
                with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
                    fname = [n for n in zf.namelist() if n.endswith(".txt")][0]
                    raw = zf.read(fname).decode("latin-1")
                with open(cache_path, "w") as f:
                    f.write(raw)
            except Exception as e:
                logger.warning(f"COT download failed: {e}")
                if not os.path.exists(cache_path):
                    raise

        with open(cache_path) as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("CFTC_Market_Code", "").strip() == market_code]

        if not rows:
            raise ValueError(f"No COT rows for market code {market_code}")

        # Most recent row
        row = rows[-1]
        comm_long = float(row.get("Comm_Positions_Long_All", 0))
        comm_short = float(row.get("Comm_Positions_Short_All", 0))
        oi = float(row.get("Open_Interest_All", 1))

        net_comm = (comm_long - comm_short) / (oi + 1e-10)
        # net_comm: positive = commercials net long (bullish for commodity)
        # For financials, invert (commercials hedge, non-comm is signal)
        is_financial = symbol in ("ES", "NQ", "EUR", "GBP", "JPY", "BTC")
        value = float(np.clip(-net_comm * 3 if is_financial else net_comm * 3, -1, 1))

        return RawSignal(
            scale="INSTITUTIONAL", source="cot",
            symbol=symbol, value=value, confidence=0.80,
            evidence_level="ESTABLISHED",
            timestamp=datetime.utcnow(),
            raw={"net_comm": net_comm, "comm_long": comm_long,
                 "comm_short": comm_short, "oi": oi},
        )


class InstitutionalFlowSignal:
    """
    Institutional flow proxy from price/volume divergence at key levels.
    In absence of dark pool data (requires commercial feed),
    uses smart money index approximation:
    — early session (first 30m) driven by retail/emotional
    — late session (last 30m) driven by institutional
    Divergence between them = institutional bias.
    [TESTABLE] — smart money index as institutional proxy.
    """

    def compute(self, bars: List[OHLCV], symbol: str) -> RawSignal:
        if len(bars) < 50:
            return RawSignal(
                scale="INSTITUTIONAL", source="inst_flow",
                symbol=symbol, value=0.0, confidence=0.1,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            )

        closes = np.array([b.close for b in bars])

        # 50-day trend as institutional baseline
        trend_50 = (closes[-1] - closes[-50]) / (closes[-50] + 1e-10)

        # 200-day trend as sovereign/secular baseline
        if len(closes) >= 200:
            trend_200 = (closes[-1] - closes[-200]) / (closes[-200] + 1e-10)
        else:
            trend_200 = trend_50

        # Institutional signal: 50-day trend vs 200-day trend
        # Both pointing same direction = institutional confirming secular
        if trend_50 * trend_200 > 0:  # same sign
            value = float(np.clip((trend_50 + trend_200) / 2 * 10, -1, 1))
        else:
            # Divergence = institutional transitioning
            value = float(np.clip(trend_50 * 5, -1, 1))

        return RawSignal(
            scale="INSTITUTIONAL", source="inst_flow",
            symbol=symbol, value=value, confidence=0.55,
            evidence_level="TESTABLE",
            timestamp=bars[-1].timestamp,
            raw={"trend_50": trend_50, "trend_200": trend_200},
            notes="Smart money proxy via 50/200d trend divergence [TESTABLE]"
        )


class CreditSpreadSignal:
    """
    Credit spread as institutional risk appetite indicator.
    Tight spreads = institutional confidence (bullish equities).
    Widening spreads = institutional risk-off (bearish equities).
    Uses HYG/LQD ratio as proxy for high-yield credit spreads.
    [ESTABLISHED] — credit as leading indicator of equity risk.
    """

    def compute(self, symbol: str = "SPY") -> RawSignal:
        try:
            import yfinance as yf
            hyg = yf.Ticker("HYG").history(period="3mo")["Close"]
            lqd = yf.Ticker("LQD").history(period="3mo")["Close"]

            if hyg.empty or lqd.empty:
                raise ValueError("No credit data")

            ratio = hyg / lqd
            ratio_current = ratio.iloc[-1]
            ratio_mean = ratio.mean()
            ratio_std = ratio.std() + 1e-10

            # Z-score: high ratio = tight spread = risk-on = bullish
            z = (ratio_current - ratio_mean) / ratio_std
            value = float(np.clip(z / 2, -1, 1))

            return RawSignal(
                scale="INSTITUTIONAL", source="credit_spread",
                symbol=symbol, value=value, confidence=0.72,
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(),
                raw={"hyg_lqd_ratio": ratio_current, "z_score": z},
            )
        except Exception as e:
            logger.warning(f"Credit spread fetch failed: {e}")
            return RawSignal(
                scale="INSTITUTIONAL", source="credit_spread",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="ESTABLISHED", timestamp=datetime.utcnow(),
            )


# ─── SOVEREIGN LAYER ──────────────────────────────────────────────────────────

class MacroSignal:
    """
    Macro indicators from FRED (Federal Reserve Economic Data — free API).
    Covers: yield curve, employment, inflation, M2 money supply.
    [ESTABLISHED] — macro as sovereign field conditions.

    API key: free at https://fred.stlouisfed.org/docs/api/api_key.html
    Set FRED_API_KEY environment variable.
    """

    FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

    SERIES = {
        "yield_curve": "T10Y2Y",    # 10yr-2yr spread (recession proxy)
        "inflation":   "CPIAUCSL",  # CPI
        "employment":  "ICSA",      # Initial jobless claims (inverted)
        "m2":          "M2SL",      # Money supply
        "dxy":         "DTWEXBGS",  # Trade-weighted USD index
    }

    def __init__(self):
        self.api_key = os.environ.get("FRED_API_KEY", "")

    def compute(self, symbol: str = "MACRO") -> RawSignal:
        if not self.api_key:
            logger.warning("FRED_API_KEY not set — sovereign macro signal unavailable")
            return RawSignal(
                scale="SOVEREIGN", source="macro_fred",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(),
                notes="Set FRED_API_KEY env var for macro signal"
            )
        try:
            scores = {}
            for name, series_id in self.SERIES.items():
                val = self._fetch_latest(series_id)
                if val is not None:
                    scores[name] = val

            if not scores:
                raise ValueError("No FRED data retrieved")

            # Yield curve: positive = normal (bullish), negative = inverted (bearish)
            yc_norm = float(np.clip(scores.get("yield_curve", 0) / 1.5, -1, 1))

            # Jobless claims: lower = better (invert and normalise)
            # ~200k = good, ~400k = bad
            claims = scores.get("employment", 300)
            claims_norm = float(np.clip(-(claims - 250000) / 200000, -1, 1))

            # Aggregate: yield curve is primary sovereign signal
            value = float(np.clip(yc_norm * 0.5 + claims_norm * 0.3, -1, 1))

            return RawSignal(
                scale="SOVEREIGN", source="macro_fred",
                symbol=symbol, value=value, confidence=0.78,
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(),
                raw=scores,
            )
        except Exception as e:
            logger.warning(f"FRED macro fetch failed: {e}")
            return RawSignal(
                scale="SOVEREIGN", source="macro_fred",
                symbol=symbol, value=0.0, confidence=0.1,
                evidence_level="ESTABLISHED", timestamp=datetime.utcnow(),
            )

    def _fetch_latest(self, series_id: str) -> Optional[float]:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": "1",
            "sort_order": "desc",
        }
        try:
            resp = requests.get(self.FRED_BASE, params=params, timeout=8)
            data = resp.json()
            obs = data.get("observations", [])
            if obs:
                val = obs[0].get("value", ".")
                return float(val) if val != "." else None
        except Exception as e:
            logger.warning(f"FRED {series_id}: {e}")
        return None


class CentralBankSignal:
    """
    Central bank stance from yield spreads and rate expectations.
    Uses 2-year treasury yield as real-time CB stance proxy.
    2yr: rising = tightening (bearish risk assets)
         falling = easing (bullish risk assets)
    [ESTABLISHED] — short-end rates as CB expectations.
    """

    def compute(self, symbol: str = "MACRO") -> RawSignal:
        try:
            import yfinance as yf
            # 2yr yield proxy via SHY ETF (short treasury)
            # Or direct: ^IRX (13-week T-bill)
            irx = yf.Ticker("^IRX").history(period="6mo")["Close"]
            if irx.empty:
                raise ValueError("No rate data")

            # Rate change over 3 months (trend of CB stance)
            rate_now = irx.iloc[-1]
            rate_3m = irx.iloc[-63] if len(irx) >= 63 else irx.iloc[0]
            rate_change = rate_now - rate_3m

            # Rising rates = tightening = bearish for risk (negative signal)
            value = float(np.clip(-rate_change / 1.5, -1, 1))

            return RawSignal(
                scale="SOVEREIGN", source="central_bank",
                symbol=symbol, value=value, confidence=0.75,
                evidence_level="ESTABLISHED",
                timestamp=datetime.utcnow(),
                raw={"rate_now": rate_now, "rate_3m": rate_3m,
                     "rate_change": rate_change},
            )
        except Exception as e:
            logger.warning(f"CB signal failed: {e}")
            return RawSignal(
                scale="SOVEREIGN", source="central_bank",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="ESTABLISHED", timestamp=datetime.utcnow(),
            )


class GeopoliticalSignal:
    """
    Geopolitical stress proxy from VIX term structure and gold/USD relationship.
    VIX contango = calm (low geo stress)
    VIX backwardation = acute stress (high geo stress)
    Gold/USD divergence = sovereign uncertainty loading.
    [TESTABLE] — VIX structure as geopolitical stress proxy.
    """

    def compute(self, symbol: str = "MACRO") -> RawSignal:
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX").history(period="2mo")["Close"]
            gold = yf.Ticker("GC=F").history(period="2mo")["Close"]
            usd = yf.Ticker("DX-Y.NYB").history(period="2mo")["Close"]

            if vix.empty or gold.empty:
                raise ValueError("Insufficient data")

            # VIX trend (rising = stress)
            vix_trend = (vix.iloc[-1] - vix.iloc[-20]) / (vix.iloc[-20] + 1e-10)

            # Gold/USD divergence (gold up + USD down = stress; gold up + USD up = extreme stress)
            gold_ret = (gold.iloc[-1] - gold.iloc[-20]) / (gold.iloc[-20] + 1e-10)
            usd_ret = (usd.iloc[-1] - usd.iloc[-20]) / (usd.iloc[-20] + 1e-10) if not usd.empty else 0

            # Stress score: high VIX trend + rising gold = geopolitical loading
            stress = vix_trend * 0.6 + gold_ret * 0.4
            value = float(np.clip(-stress * 5, -1, 1))  # inverted: stress = bearish

            return RawSignal(
                scale="SOVEREIGN", source="geopolitical",
                symbol=symbol, value=value, confidence=0.55,
                evidence_level="TESTABLE",
                timestamp=datetime.utcnow(),
                raw={"vix_trend": vix_trend, "gold_ret": gold_ret, "usd_ret": usd_ret},
                notes="VIX + gold/USD proxy [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"Geo signal failed: {e}")
            return RawSignal(
                scale="SOVEREIGN", source="geopolitical",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.utcnow(),
            )
