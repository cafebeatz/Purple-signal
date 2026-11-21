"""Purple Signal — fluxo como direção principal, tendência e KDJ como confirmações.
A leitura de fluxo é uma estimativa baseada em OHLCV (corpo/range ponderado por volume),
pois a integração atual não fornece agressão bid/ask nem livro de ofertas.
"""
from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(series, fast) - ema(series, slow)
    signal_line = ema(line, signal)
    return line, signal_line, line - signal_line


def didi_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    return plus_di.fillna(0), minus_di.fillna(0)


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9):
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    spread = (highest - lowest).replace(0, np.nan)
    rsv = (100 * (close - lowest) / spread).fillna(50)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def _estimated_flow(df: pd.DataFrame, window: int = 12) -> float:
    """Pressão aproximada: candle positivo/negativo ponderado pelo volume."""
    spread = (df.high - df.low).replace(0, np.nan)
    pressure = ((df.close - df.open) / spread).clip(-1, 1).fillna(0)
    volume = pd.to_numeric(df.get("volume", pd.Series(0, index=df.index)), errors="coerce").fillna(0).clip(lower=0)
    weights = volume.tail(window)
    values = pressure.tail(window)
    if float(weights.sum()) <= 0:
        return float(values.mean())
    return float((values * weights).sum() / weights.sum())


def _hidden_structure_confluence(df: pd.DataFrame, direction: str, atr_value: float) -> tuple[int, str]:
    """Confluência privada de pivôs/suporte/resistência. Não retorna níveis ao painel."""
    if len(df) < 12 or not np.isfinite(atr_value) or atr_value <= 0:
        return 0, "NONE"
    highs = df.high.astype(float).to_numpy()
    lows = df.low.astype(float).to_numpy()
    piv_hi, piv_lo = [], []
    # Pivô confirmado: duas velas à esquerda e duas à direita.
    for i in range(max(2, len(df) - 45), len(df) - 2):
        if highs[i] >= max(highs[i-2:i]) and highs[i] > max(highs[i+1:i+3]):
            piv_hi.append(highs[i])
        if lows[i] <= min(lows[i-2:i]) and lows[i] < min(lows[i+1:i+3]):
            piv_lo.append(lows[i])
    price = float(df.close.iloc[-1])
    tolerance = max(atr_value * 0.35, abs(price) * 0.00005)
    near_support = bool(piv_lo and min(abs(price - x) for x in piv_lo[-8:]) <= tolerance)
    near_resistance = bool(piv_hi and min(abs(price - x) for x in piv_hi[-8:]) <= tolerance)
    prior_high = max(piv_hi[-8:]) if piv_hi else None
    prior_low = min(piv_lo[-8:]) if piv_lo else None
    last = df.iloc[-1]
    avg_vol = float(pd.to_numeric(df.volume, errors="coerce").fillna(0).tail(20).mean()) if "volume" in df else 0
    vol = float(last.get("volume", 0) or 0)
    breakout_up = prior_high is not None and price > prior_high + tolerance * 0.15 and (avg_vol <= 0 or vol >= avg_vol * 1.15)
    breakout_down = prior_low is not None and price < prior_low - tolerance * 0.15 and (avg_vol <= 0 or vol >= avg_vol * 1.15)
    if direction == "CALL":
        if near_support or breakout_up:
            return 8, "SUPPORT_OR_BREAKOUT"
        if near_resistance:
            return -8, "RESISTANCE"
    elif direction == "PUT":
        if near_resistance or breakout_down:
            return 8, "RESISTANCE_OR_BREAKDOWN"
        if near_support:
            return -8, "SUPPORT"
    return 0, "NONE"


Direction = Literal["CALL", "PUT", "NEUTRO"]


def analyze_candles(candles: list[dict]) -> dict:
    """Analisa somente a lista recebida; o servidor entrega aqui apenas velas fechadas."""
    if len(candles) < 50:
        return {"ready": False, "reason": "Candles insuficientes"}
    df = pd.DataFrame(candles).copy()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if len(df) < 50:
        return {"ready": False, "reason": "Candles válidos insuficientes"}

    close, high, low = df.close, df.high, df.low
    ma9, ma21 = ema(close, 9), ema(close, 21)
    macd_line, macd_signal, macd_hist = macd(close)
    rsi14 = rsi(close, 14)
    plus_di, minus_di = didi_index(high, low, close, 14)
    k, d, j = kdj(high, low, close, 9)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    i, ip = len(df)-1, len(df)-2

    # 1) FLUXO é a autoridade direcional. É uma estimativa OHLCV, não tape reading.
    flow = _estimated_flow(df)
    flow_direction = "CALL" if flow >= 0.08 else ("PUT" if flow <= -0.08 else "NEUTRO")
    flow_strength = min(100, int(abs(flow) * 100))

    # 2) Tendência precisa acompanhar o fluxo; médias são confirmação, não gatilho soberano.
    ma_bull = ma9.iloc[i] > ma21.iloc[i] and ma9.iloc[i] > ma9.iloc[i-3]
    ma_bear = ma9.iloc[i] < ma21.iloc[i] and ma9.iloc[i] < ma9.iloc[i-3]
    trend = "CALL" if ma_bull else ("PUT" if ma_bear else "NEUTRO")

    # 3) KDJ reforça a direção; não pode inverter fluxo/tendência.
    k_now, d_now, j_now = float(k.iloc[i]), float(d.iloc[i]), float(j.iloc[i])
    kdj_bull = k_now > d_now and j_now > float(j.iloc[ip])
    kdj_bear = k_now < d_now and j_now < float(j.iloc[ip])
    kdj_direction = "CALL" if kdj_bull else ("PUT" if kdj_bear else "NEUTRO")

    direction: Direction = "NEUTRO"
    if flow_direction != "NEUTRO" and flow_direction == trend and flow_direction == kdj_direction:
        direction = flow_direction

    # 4) Indicadores de entrada: duas confirmações dentre MACD, RSI e DI.
    macd_bull = float(macd_line.iloc[i]) > float(macd_signal.iloc[i])
    macd_bear = float(macd_line.iloc[i]) < float(macd_signal.iloc[i])
    rsi_value = float(rsi14.iloc[i])
    rsi_bull = 50 < rsi_value < 75
    rsi_bear = 25 < rsi_value < 50
    dp, dm = float(plus_di.iloc[i]), float(minus_di.iloc[i])
    di_bull, di_bear = dp > dm, dm > dp
    confirms = [macd_bull, rsi_bull, di_bull] if direction == "CALL" else (
        [macd_bear, rsi_bear, di_bear] if direction == "PUT" else []
    )
    confirm_count = sum(confirms)
    if direction != "NEUTRO" and confirm_count < 2:
        direction = "NEUTRO"

    atr_value = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
    structure_bonus, _private_structure_tag = _hidden_structure_confluence(df, direction, atr_value)
    confidence = 0
    if direction != "NEUTRO":
        confidence = min(97, 62 + max(0, flow_strength - 8) // 3 + confirm_count * 5 + structure_bonus)
        confidence = max(50, confidence)

    macd_cross_up = float(macd_line.iloc[ip]) <= float(macd_signal.iloc[ip]) and macd_bull
    macd_cross_down = float(macd_line.iloc[ip]) >= float(macd_signal.iloc[ip]) and macd_bear
    di_cross_up = float(plus_di.iloc[ip]) <= float(minus_di.iloc[ip]) and di_bull
    di_cross_down = float(plus_di.iloc[ip]) >= float(minus_di.iloc[ip]) and di_bear
    return {
        "ready": True,
        "direction": direction,
        "confidence": int(confidence),
        "price": float(close.iloc[i]),
        "indicators": {
            "flow_direction": flow_direction,
            "flow_pressure": round(flow, 3),
            "flow_strength": flow_strength,
            "trend_direction": trend,
            "kdj_k": round(k_now, 2),
            "kdj_d": round(d_now, 2),
            "kdj_j": round(j_now, 2),
            "kdj_direction": kdj_direction,
            "ema9": round(float(ma9.iloc[i]), 6),
            "ema21": round(float(ma21.iloc[i]), 6),
            "macd": round(float(macd_line.iloc[i]), 6),
            "macd_signal": round(float(macd_signal.iloc[i]), 6),
            "macd_status": "COMPRA" if macd_cross_up else ("VENDA" if macd_cross_down else ("ALTA" if macd_bull else "BAIXA")),
            "rsi": round(rsi_value, 2),
            "rsi_status": "COMPRA" if rsi_bull else ("VENDA" if rsi_bear else "NEUTRO"),
            "di_plus": round(dp, 2),
            "di_minus": round(dm, 2),
            "didi_status": "AGULHADA COMPRA" if di_cross_up else ("AGULHADA VENDA" if di_cross_down else ("DI+ > DI-" if di_bull else "DI- > DI+")),
            "entry_confirmations": int(confirm_count),
            "flow_source": "ESTIMADO_OHLCV",
        },
    }
