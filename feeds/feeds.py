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
import json
import hashlib
import requests
import numpy as np
from datetime import datetime, timedelta, timezone
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
                timestamp=datetime.now(timezone.utc),
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
                timestamp=datetime.now(timezone.utc),
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
                timestamp=datetime.now(timezone.utc),
                raw={"vix": vix},
                notes="VIX proxy — inverse: high VIX = bearish participant sentiment"
            )
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")
            return RawSignal(
                scale="PARTICIPANT", source="options_skew",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc),
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
                timestamp=datetime.now(timezone.utc),
                raw={"pc_ratio": pc_ratio, "call_vol": call_vol, "put_vol": put_vol},
            )
        except Exception as e:
            logger.warning(f"Options skew failed for {symbol}: {e}")
            return RawSignal(
                scale="PARTICIPANT", source="options_skew",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc),
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
                timestamp=datetime.now(timezone.utc),
                notes=f"No COT mapping for {symbol}"
            )
        try:
            return self._fetch_cot(symbol, market_code)
        except Exception as e:
            logger.warning(f"COT fetch failed for {symbol}: {e}")
            return RawSignal(
                scale="INSTITUTIONAL", source="cot",
                symbol=symbol, value=0.0, confidence=0.2,
                evidence_level="ESTABLISHED", timestamp=datetime.now(timezone.utc),
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
            timestamp=datetime.now(timezone.utc),
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
                evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc),
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
                timestamp=datetime.now(timezone.utc),
                raw={"hyg_lqd_ratio": ratio_current, "z_score": z},
            )
        except Exception as e:
            logger.warning(f"Credit spread fetch failed: {e}")
            return RawSignal(
                scale="INSTITUTIONAL", source="credit_spread",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="ESTABLISHED", timestamp=datetime.now(timezone.utc),
            )


# ─── SOVEREIGN LAYER ──────────────────────────────────────────────────────────


# ─── FRED Cache Helper ──────────────────────────────────────────────────────

FRED_CACHE_DIR = "/tmp/bmr_fred_cache"
os.makedirs(FRED_CACHE_DIR, exist_ok=True)

def _fred_request_cached(url: str, params: Dict[str, Any], max_age_hours: float = 24) -> Optional[Dict]:
    """FRED API request with file-based caching."""
    # Ensure api_key is not in cache key (it's sensitive and constant)
    cache_params = params.copy()
    cache_params.pop("api_key", None)
    
    # Sort keys to ensure consistent hashing
    param_str = json.dumps(cache_params, sort_keys=True)
    cache_key = hashlib.md5(f"{url}{param_str}".encode()).hexdigest()
    cache_path = os.path.join(FRED_CACHE_DIR, f"{cache_key}.json")
    
    # Check cache
    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < max_age_hours * 3600:
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
                
    # Fetch from API
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            try:
                with open(cache_path, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass
            return data
        elif resp.status_code == 429:
            logger.error("FRED API rate limit hit!")
        else:
            logger.warning(f"FRED API error {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"FRED API request failed: {e}")
    return None


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
        # ── Monetary regime ──────────────────────────────────────────────────
        "yield_curve":    "T10Y2Y",       # 10yr-2yr spread (recession proxy)
        "fed_funds":      "FEDFUNDS",     # Effective fed funds rate
        "breakeven_5y":   "T5YIE",        # 5-year inflation breakeven
        # ── Inflation ────────────────────────────────────────────────────────
        "cpi":            "CPIAUCSL",     # CPI all items
        "pce":            "PCEPI",        # PCE price index (Fed's preferred measure)
        # ── Labor market ─────────────────────────────────────────────────────
        "initial_claims": "ICSA",         # Initial jobless claims (weekly, leading)
        "unemployment":   "UNRATE",       # Unemployment rate
        # ── Money supply / liquidity ─────────────────────────────────────────
        "m2":             "M2SL",         # M2 money supply
        "dxy":            "DTWEXBGS",     # Trade-weighted USD index
        # ── Credit conditions ────────────────────────────────────────────────
        "hy_spread":      "BAMLH0A0HYM2", # ICE BofA HY credit spread (OAS)
        "mortgage_rate":  "MORTGAGE30US", # 30yr fixed mortgage rate
        # ── Community / consumer wellbeing (KindPath Layer 2) ────────────────
        "sentiment":      "UMCSENT",      # U of Michigan consumer sentiment
        "savings_rate":   "PSAVERT",      # Personal savings rate
        "real_income":    "DSPIC96",      # Real disposable personal income per capita
        # ── Australian context ────────────────────────────────────────────────
        "aud_usd":        "DEXAUS",       # AUD/USD exchange rate
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
                timestamp=datetime.now(timezone.utc),
                notes="Set FRED_API_KEY env var for macro signal"
            )
        try:
            # Check for updates today (1-hour cached check)
            force_refresh = False
            try:
                meta = FREDMetaSignal()
                if meta._check_updates_today():
                    force_refresh = True
                    logger.info("FRED updates detected today — forcing MacroSignal refresh")
            except Exception:
                pass
            
            cache_hours = 0 if force_refresh else 24

            readings: Dict[str, list] = {}
            for name, series_id in self.SERIES.items():
                obs = self._fetch_observations(series_id, limit=2, cache_hours=cache_hours)
                if obs:
                    readings[name] = obs

            if not readings:
                raise ValueError("No FRED data retrieved")

            scores: Dict[str, float] = {}

            # ── Monetary regime ───────────────────────────────────────────────
            # Yield curve: +1.5pp = fully normal, -1.5pp = deeply inverted
            if "yield_curve" in readings:
                v = readings["yield_curve"][0][1]
                scores["yield_curve"] = float(np.clip(v / 1.5, -1, 1))

            # Fed funds: 0%→+0.5 (easing), 3%→0 (neutral), 7%→-1 (tight)
            if "fed_funds" in readings:
                v = readings["fed_funds"][0][1]
                scores["fed_funds"] = float(np.clip(-(v - 3.0) / 4.0, -1, 1))

            # 5yr breakeven: 2% = optimal; penalise extremes
            if "breakeven_5y" in readings:
                v = readings["breakeven_5y"][0][1]
                deviation = abs(v - 2.0)
                scores["breakeven_5y"] = float(np.clip(0.5 - deviation / 2.0, -1, 1))

            # ── Inflation ─────────────────────────────────────────────────────
            # CPI: use 2-period trend (annualised monthly change)
            if "cpi" in readings and len(readings["cpi"]) >= 2:
                v_now, v_prior = readings["cpi"][0][1], readings["cpi"][1][1]
                ann_change = (v_now - v_prior) / (v_prior + 1e-10) * 12 * 100
                # 2% = neutral, <1% = deflation risk, >5% = inflation risk
                scores["cpi"] = float(np.clip(-(ann_change - 2.0) / 3.5, -1, 1))

            # PCE: same approach (Fed's preferred measure)
            if "pce" in readings and len(readings["pce"]) >= 2:
                v_now, v_prior = readings["pce"][0][1], readings["pce"][1][1]
                ann_change = (v_now - v_prior) / (v_prior + 1e-10) * 12 * 100
                scores["pce"] = float(np.clip(-(ann_change - 2.0) / 3.5, -1, 1))

            # ── Labor market ──────────────────────────────────────────────────
            # Initial claims: ~200k = good, ~400k = severe stress
            if "initial_claims" in readings:
                v = readings["initial_claims"][0][1]
                scores["initial_claims"] = float(np.clip(-(v - 250000) / 200000, -1, 1))

            # Unemployment rate: 3.5%→+0.5, 6%→0, 10%→-1
            if "unemployment" in readings:
                v = readings["unemployment"][0][1]
                scores["unemployment"] = float(np.clip(-(v - 4.5) / 4.0, -1, 1))

            # ── Credit conditions ─────────────────────────────────────────────
            # HY credit spread (OAS %): 2%→+1, 4%→0, 8%→-1
            if "hy_spread" in readings:
                v = readings["hy_spread"][0][1]
                scores["hy_spread"] = float(np.clip(-(v - 3.5) / 3.5, -1, 1))

            # Mortgage rate: 3%→+0.5, 5.5%→0, 9%→-1
            if "mortgage_rate" in readings:
                v = readings["mortgage_rate"][0][1]
                scores["mortgage_rate"] = float(np.clip(-(v - 5.5) / 3.0, -1, 1))

            # ── Community / consumer wellbeing ────────────────────────────────
            # Consumer sentiment: 90+=bullish, 70=neutral, 50-=bearish
            if "sentiment" in readings:
                v = readings["sentiment"][0][1]
                scores["sentiment"] = float(np.clip((v - 70) / 25, -1, 1))

            # Savings rate: 6–10% = healthy; <2% = depleted; >15% = fear-hoarding
            if "savings_rate" in readings:
                v = readings["savings_rate"][0][1]
                if v < 6:
                    scores["savings_rate"] = float(np.clip((v - 3) / 5, -1, 0.3))
                elif v > 12:
                    scores["savings_rate"] = float(np.clip(-(v - 12) / 6, -0.5, 0.2))
                else:
                    scores["savings_rate"] = 0.3  # healthy zone

            # Real income: direction of change (trend signal)
            if "real_income" in readings and len(readings["real_income"]) >= 2:
                v_now = readings["real_income"][0][1]
                v_prior = readings["real_income"][1][1]
                pct_change = (v_now - v_prior) / (abs(v_prior) + 1e-10)
                scores["real_income"] = float(np.clip(pct_change * 50, -1, 1))

            # AUD/USD: trend-based (falling DEXAUS = strengthening AUD = risk-on = bullish)
            if "aud_usd" in readings and len(readings["aud_usd"]) >= 2:
                v_now   = readings["aud_usd"][0][1]
                v_prior = readings["aud_usd"][1][1]
                pct_chg = (v_now - v_prior) / (v_prior + 1e-10)
                # Weakening AUD (rising DEXAUS) = bearish for risk assets
                scores["aud_usd"] = float(np.clip(-pct_chg * 20, -1, 1))

            # ── Aggregate by group with weights ───────────────────────────────
            group_weights = {
                "monetary":  (["yield_curve", "fed_funds", "breakeven_5y"], 0.30),
                "labor":     (["initial_claims", "unemployment"],            0.25),
                "credit":    (["hy_spread", "mortgage_rate"],                0.25),
                "community": (["sentiment", "savings_rate", "real_income",
                               "cpi", "pce", "aud_usd"],                   0.20),
            }

            value = 0.0
            total_weight = 0.0
            group_avgs: Dict[str, float] = {}
            for group, (series_list, weight) in group_weights.items():
                group_scores = [scores[s] for s in series_list if s in scores]
                if group_scores:
                    avg = float(np.mean(group_scores))
                    group_avgs[group] = avg
                    value += avg * weight
                    total_weight += weight

            if total_weight > 0:
                value = value / total_weight

            value = float(np.clip(value, -1, 1))

            # Confidence scales with data coverage
            coverage = len(scores) / len(self.SERIES)
            confidence = round(0.40 + 0.45 * coverage, 3)

            # ── FRED metadata layer ───────────────────────────────────────────
            fred_meta: Dict[str, Any] = {}
            try:
                meta_signal = FREDMetaSignal()
                fred_meta = meta_signal.compute()

                # Pre-release window: data may be about to be superseded
                # Reduce confidence slightly — signal is noisier before a release
                if fred_meta.get("pre_release_window"):
                    confidence = round(max(0.1, confidence - 0.10), 3)
                    logger.info("FRED pre-release window active — confidence reduced")

                # Revision delta: quiet restatement detected  
                # Large absolute revision (|delta| > 0.1) = structural absence signal                                       
                revision_deltas = fred_meta.get("revision_delta", {})                                                       
                large_revisions = {                           
                    k: v for k, v in revision_deltas.items() if abs(v) > 0.1                                                
                }                                                               
                if large_revisions:                           
                    confidence = round(max(0.1, confidence - 0.05), 3)                                                      
                    logger.info(f"FRED quiet restatement detected: {large_revisions}")                                      

                # Series update monitoring: Was any core series updated in the last 24h?
                # This is a high-significance signal for "freshness" and potential volatility.
                fred_meta["fred_update_today"] = any(
                    fm.get("last_updated") and 
                    (datetime.now(timezone.utc) - datetime.fromisoformat(fm["last_updated"].replace("Z", "+00:00"))).total_seconds() < 86400
                    for fm in fred_meta.get("series_metadata", {}).values()
                )
                if fred_meta["fred_update_today"]:
                    logger.info("FRED update detected within last 24h — signal freshness high")
            except Exception as e:
                logger.debug(f"FREDMetaSignal annotation failed: {e}")

            return RawSignal(
                scale="SOVEREIGN", source="macro_fred",
                symbol=symbol, value=value, confidence=confidence,
                evidence_level="ESTABLISHED",
                timestamp=datetime.now(timezone.utc),
                raw={
                    "scores": scores,
                    "group_avgs": group_avgs,
                    "coverage": f"{len(scores)}/{len(self.SERIES)}",
                    "fred_meta": fred_meta,
                },
            )
        except Exception as e:
            logger.warning(f"FRED macro fetch failed: {e}")
            return RawSignal(
                scale="SOVEREIGN", source="macro_fred",
                symbol=symbol, value=0.0, confidence=0.1,
                evidence_level="ESTABLISHED", timestamp=datetime.now(timezone.utc),
            )

    def _fetch_observations(self, series_id: str, limit: int = 2,
                            cache_hours: int = 24,
                            observation_start: Optional[str] = None,
                            observation_end: Optional[str] = None) -> list:
        """
        Return up to `limit` observations for series_id, newest first.
        Each element is a (date_str, float_value) tuple.
        Optionally filter by observation_start / observation_end (YYYY-MM-DD).
        """
        params: Dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": str(limit),
            "sort_order": "desc",
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        data = _fred_request_cached(self.FRED_BASE, params, max_age_hours=cache_hours)
        if data:
            result = []
            for o in data.get("observations", []):
                val = o.get("value", ".")
                if val != ".":
                    result.append((o["date"], float(val)))
            return result
        return []


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
                timestamp=datetime.now(timezone.utc),
                raw={"rate_now": rate_now, "rate_3m": rate_3m,
                     "rate_change": rate_change},
            )
        except Exception as e:
            logger.warning(f"CB signal failed: {e}")
            return RawSignal(
                scale="SOVEREIGN", source="central_bank",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="ESTABLISHED", timestamp=datetime.now(timezone.utc),
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
                timestamp=datetime.now(timezone.utc),
                raw={"vix_trend": vix_trend, "gold_ret": gold_ret, "usd_ret": usd_ret},
                notes="VIX + gold/USD proxy [TESTABLE]"
            )
        except Exception as e:
            logger.warning(f"Geo signal failed: {e}")
            return RawSignal(
                scale="SOVEREIGN", source="geopolitical",
                symbol=symbol, value=0.0, confidence=0.0,
                evidence_level="TESTABLE", timestamp=datetime.now(timezone.utc),
            )


# ─── FRED Metadata Layer ─────────────────────────────────────────────────────

class FREDMetaSignal:
    """
    Extended FRED endpoint integration: vintage dates, release timing,
    series update detection, and NSW regional data.

    Produces a metadata dict for the SignalLogger (not a RawSignal).
    These are signal context variables, not market signals themselves.

    Endpoints:
      fred/series/vintagedates — revision history (data was quietly restated)
      fred/releases/dates     — next scheduled release date
      fred/series/updates     — recently updated series
      fred/series/observations (regional) — NSW vs national unemployment

    [TESTABLE] — revision delta as absence signal requires calibration.
    [ESTABLISHED] — release timing windows are factual schedule data.
    """

    FRED_BASE = "https://api.stlouisfed.org/fred"

    # Core series that trigger pre-release window flags
    CORE_SERIES = ["T10Y2Y", "BAMLH0A0HYM2", "FEDFUNDS", "CPIAUCSL",
                   "M2SL", "DEXAUS", "UNRATE"]

    # NSW regional series (if available via FRED Maps API)
    # FRED regional series codes for NSW unemployment proxy
    # Note: FRED has limited Australian regional data; using national as fallback
    NSW_SERIES = "LRHUTTTTAUA156S"  # Australia unemployment rate (monthly)

    def __init__(self):
        self.api_key = os.environ.get("FRED_API_KEY", "")

    def compute(self) -> Dict[str, Any]:
        """
        Returns fred_meta dict with all extended FRED signals.
        Safe — never raises; returns partial dict on failures.
        """
        meta: Dict[str, Any] = {}
        if not self.api_key:
            return meta

        try:
            meta["revision_delta"]           = self._vintage_revision_deltas()
            meta["pre_release_window"]       = self._check_pre_release_window()
            meta["fred_update_today"]        = self._check_updates_today()
            meta["regional_vs_national_gap"] = self._regional_vs_national()
        except Exception as e:
            logger.warning(f"FREDMetaSignal.compute failed: {e}")
        return meta

    # ── 1b. Vintage dates — revision history signal ───────────────────────

    def _vintage_revision_deltas(self) -> Dict[str, float]:
        """
        For each core series: current value minus value at 3 prior vintage dates.
        Large delta = data was quietly restated (absence signal).
        """
        deltas: Dict[str, float] = {}
        for series_id in self.CORE_SERIES[:4]:  # top 4 most frequently revised
            try:
                # Fetch last 3 vintage dates
                vd_url = f"{self.FRED_BASE}/series/vintagedates"
                params = {
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "limit": "3",
                    "sort_order": "desc",
                }
                data = _fred_request_cached(vd_url, params, max_age_hours=24)
                if not data:
                    continue
                    
                vintage_dates = [
                    v["vintage_date"]
                    for v in data.get("vintage_dates", [])
                ]
                if len(vintage_dates) < 2:
                    continue

                # Current value
                current = self._fetch_latest_obs(series_id)
                # Value at oldest vintage
                vintage_val = self._fetch_obs_at_vintage(series_id, vintage_dates[-1])

                if current is not None and vintage_val is not None:
                    deltas[series_id] = round(current - vintage_val, 6)

            except Exception as e:
                logger.debug(f"Vintage dates for {series_id}: {e}")

        return deltas

    # ── 1c. Release timing signal ─────────────────────────────────────────

    def _check_pre_release_window(self, days_threshold: int = 3) -> bool:
        """
        Returns True if any core series has a scheduled release within
        `days_threshold` days. Signals near release carry different weight.
        """
        try:
            today = datetime.now(timezone.utc).date()
            url = f"{self.FRED_BASE}/releases/dates"
            params = {
                "api_key": self.api_key,
                "file_type": "json",
                "limit": "50",
                "sort_order": "asc",
                "include_release_dates_with_no_data": "false",
            }
            # Release dates change daily
            data = _fred_request_cached(url, params, max_age_hours=12)
            if not data:
                return False
                
            release_dates = data.get("release_dates", [])
            for rd in release_dates:
                try:
                    rel_date = datetime.strptime(rd["date"], "%Y-%m-%d").date()
                    days_to  = (rel_date - today).days
                    if 0 <= days_to <= days_threshold:
                        return True
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"Release dates check failed: {e}")
        return False

    # ── 1d. Series updates monitor ────────────────────────────────────────

    def _check_updates_today(self) -> bool:
        """
        Returns True if any of the 7 core series were updated in the last 24h.
        If updated: MacroSignal should bypass cache and pull fresh data.
        """
        try:
            url = f"{self.FRED_BASE}/series/updates"
            params = {
                "api_key": self.api_key,
                "file_type": "json",
                "limit": "100",
                "filter_value": "macro",
            }
            # Short cache for update check
            data = _fred_request_cached(url, params, max_age_hours=1)
            if not data:
                return False
                
            updated_ids = {
                s["id"] for s in data.get("seriess", [])
            }
            return bool(updated_ids & set(self.CORE_SERIES))
        except Exception as e:
            logger.debug(f"FRED updates check failed: {e}")
        return False

    # ── 1e. Regional vs national gap ─────────────────────────────────────

    def _regional_vs_national(self) -> Optional[float]:
        """
        NSW unemployment vs national unemployment.
        Positive = NSW worse than national average.

        FRED has limited Australian regional data.
        Uses national Australian unemployment as both values with a small
        jitter acknowledgement where regional data is unavailable.
        Phase 2: replace with ABS regional data feed when available.
        [TESTABLE]
        """
        try:
            # Use MacroSignal SERIES keys for consistency
            national = self._fetch_latest_obs("UNRATE")  # US national as proxy
            aus_nat  = self._fetch_latest_obs(self.NSW_SERIES)
            if national is not None and aus_nat is not None:
                return round(aus_nat - national, 4)
        except Exception as e:
            logger.debug(f"Regional gap computation failed: {e}")
        return None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _fetch_latest_obs(self, series_id: str) -> Optional[float]:
        try:
            url = f"{self.FRED_BASE}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": "1",
                "sort_order": "desc",
            }
            data = _fred_request_cached(url, params, max_age_hours=24)
            if data:
                obs = data.get("observations", [])
                if obs:
                    val = obs[0].get("value", ".")
                    return float(val) if val != "." else None
        except Exception:
            pass
        return None

    def fetch_historical(
        self,
        series_ids: Optional[list] = None,
        years: int = 5,
        limit_per_series: int = 260,
    ) -> Dict[str, list]:
        """
        Pull historical observations for a set of series.
        Used by the training data generator to build FRED-aware Q&A examples.

        Returns: {series_id: [(date, value), ...]} oldest-first.
        """
        if not self.api_key:
            return {}
        if series_ids is None:
            series_ids = self.CORE_SERIES

        from datetime import timedelta
        start = (datetime.now(timezone.utc) - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        result: Dict[str, list] = {}

        for sid in series_ids:
            try:
                url = f"{self.FRED_BASE}/series/observations"
                params = {
                    "series_id": sid,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "limit": str(limit_per_series),
                    "sort_order": "asc",
                    "observation_start": start,
                }
                data = _fred_request_cached(url, params, max_age_hours=24)
                if not data:
                    continue
                obs = [
                    (o["date"], float(o["value"]))
                    for o in data.get("observations", [])
                    if o.get("value", ".") != "."
                ]
                if obs:
                    result[sid] = obs
                    logger.debug(f"FRED historical: {sid} → {len(obs)} observations")
            except Exception as e:
                logger.warning(f"FRED historical fetch failed for {sid}: {e}")

        return result

    def _fetch_obs_at_vintage(
        self, series_id: str, vintage_date: str
    ) -> Optional[float]:
        try:
            url = f"{self.FRED_BASE}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": "1",
                "sort_order": "desc",
                "vintage_dates": vintage_date,
            }
            data = _fred_request_cached(url, params, max_age_hours=24)
            if data:
                obs = data.get("observations", [])
                if obs:
                    val = obs[0].get("value", ".")
                    return float(val) if val != "." else None
        except Exception:
            pass
        return None
