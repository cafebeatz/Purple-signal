"""
Estratégia Didi Fluxo MACD + RSI
- Direção: Cruzamento MA 9 x MA 21
- Gatilho: Cruzamento MACD (linha MACD x linha de sinal)
- Confirmação: RSI(14) + Agulhada do Didi (DI+ x DI-)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal


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
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def didi_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """
    Didi Index (ADX components DI+ and DI-).
    Retorna DI+ e DI- - a 'Agulhada do Didi' acontece no cruzamento de DI+ e DI-.
    """
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    mask = plus_dm > minus_dm
    plus_dm = plus_dm.where(mask, 0.0)
    minus_dm = minus_dm.where(~mask, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr)
    return plus_di.fillna(0), minus_di.fillna(0)


Direction = Literal["CALL", "PUT", "NEUTRO"]


def analyze_candles(candles: list[dict]) -> dict:
    """
    candles: lista de dicts com keys open, high, low, close, volume, ts (crescente no tempo).
    Retorna análise + sinal consolidado.
    """
    if len(candles) < 50:
        return {"ready": False, "reason": "Candles insuficientes"}

    df = pd.DataFrame(candles)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    ma9 = sma(close, 9)
    ma21 = sma(close, 21)
    macd_line, signal_line, hist = macd(close)
    rsi14 = rsi(close, 14)
    plus_di, minus_di = didi_index(high, low, close, 14)

    i = len(df) - 1
    ip = i - 1

    # 1. Direção: cruzamento MA9 x MA21
    ma_bull_cross = ma9.iloc[ip] <= ma21.iloc[ip] and ma9.iloc[i] > ma21.iloc[i]
    ma_bear_cross = ma9.iloc[ip] >= ma21.iloc[ip] and ma9.iloc[i] < ma21.iloc[i]
    ma_bull = ma9.iloc[i] > ma21.iloc[i]
    ma_bear = ma9.iloc[i] < ma21.iloc[i]
    ma_status = "ALTA" if ma_bull else ("BAIXA" if ma_bear else "LATERAL")

    # 2. Gatilho: cruzamento MACD
    macd_bull_cross = macd_line.iloc[ip] <= signal_line.iloc[ip] and macd_line.iloc[i] > signal_line.iloc[i]
    macd_bear_cross = macd_line.iloc[ip] >= signal_line.iloc[ip] and macd_line.iloc[i] < signal_line.iloc[i]
    macd_bull = macd_line.iloc[i] > signal_line.iloc[i]
    macd_bear = macd_line.iloc[i] < signal_line.iloc[i]
    macd_status = "CRUZADO COMPRA" if macd_bull_cross else (
        "CRUZADO VENDA" if macd_bear_cross else (
            "COMPRA" if macd_bull else "VENDA"
        )
    )

    # 3. Confirmação RSI
    rsi_val = float(rsi14.iloc[i])
    rsi_strong_buy = 50 < rsi_val < 70
    rsi_strong_sell = 30 < rsi_val < 50
    rsi_status = (
        "FORTE COMPRA" if rsi_strong_buy else
        "FORTE VENDA" if rsi_strong_sell else
        "SOBRECOMPRADO" if rsi_val >= 70 else
        "SOBREVENDIDO" if rsi_val <= 30 else "NEUTRO"
    )

    # 4. Agulhada do Didi (DI+ cruzando DI-)
    dp = float(plus_di.iloc[i])
    dm = float(minus_di.iloc[i])
    dp_prev = float(plus_di.iloc[ip])
    dm_prev = float(minus_di.iloc[ip])
    didi_bull_needle = dp_prev <= dm_prev and dp > dm
    didi_bear_needle = dp_prev >= dm_prev and dp < dm
    didi_bull = dp > dm
    didi_bear = dp < dm
    didi_status = (
        "AGULHADA COMPRA" if didi_bull_needle else
        "AGULHADA VENDA" if didi_bear_needle else
        "DI+ > DI-" if didi_bull else
        "DI- > DI+" if didi_bear else "NEUTRO"
    )

    # Score de confluência (0-100)
    call_votes = sum([
        ma_bull or ma_bull_cross,
        macd_bull or macd_bull_cross,
        rsi_strong_buy,
        didi_bull or didi_bull_needle,
    ])
    put_votes = sum([
        ma_bear or ma_bear_cross,
        macd_bear or macd_bear_cross,
        rsi_strong_sell,
        didi_bear or didi_bear_needle,
    ])

    direction: Direction = "NEUTRO"
    confidence = 0
    if call_votes >= 3 and call_votes > put_votes:
        direction = "CALL"
        base = 60 + (call_votes - 3) * 15
        bonus = (10 if macd_bull_cross else 0) + (10 if didi_bull_needle else 0) + (5 if ma_bull_cross else 0)
        confidence = min(99, base + bonus)
    elif put_votes >= 3 and put_votes > call_votes:
        direction = "PUT"
        base = 60 + (put_votes - 3) * 15
        bonus = (10 if macd_bear_cross else 0) + (10 if didi_bear_needle else 0) + (5 if ma_bear_cross else 0)
        confidence = min(99, base + bonus)

    return {
        "ready": True,
        "direction": direction,
        "confidence": int(confidence),
        "price": float(close.iloc[i]),
        "indicators": {
            "ma9": round(float(ma9.iloc[i]), 6),
            "ma21": round(float(ma21.iloc[i]), 6),
            "ma_status": ma_status,
            "ma_cross": "COMPRA" if ma_bull_cross else ("VENDA" if ma_bear_cross else "-"),
            "macd": round(float(macd_line.iloc[i]), 6),
            "macd_signal": round(float(signal_line.iloc[i]), 6),
            "macd_hist": round(float(hist.iloc[i]), 6),
            "macd_status": macd_status,
            "rsi": round(rsi_val, 2),
            "rsi_status": rsi_status,
            "di_plus": round(dp, 2),
            "di_minus": round(dm, 2),
            "didi_status": didi_status,
        },
    }
