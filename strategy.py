"""Purple Signal - núcleo direcional por tendência, fluxo e estrutura.

Compatível com o contrato usado pelo server.py: analyze_candles(candles)
retorna ready, direction, confidence e indicadores.
O fluxo é estimado por OHLCV; não representa agressão bid/ask real.
"""
from __future__ import annotations

from typing import Any, Iterable
import math
import pandas as pd


def sma(values, period: int):
    return pd.Series(values, dtype="float64").rolling(period).mean()


def ema(values, period: int):
    return pd.Series(values, dtype="float64").ewm(span=period, adjust=False).mean()


def rsi(values, period: int = 14):
    s = pd.Series(values, dtype="float64")
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    out = 100 - 100 / (1 + rs)
    out = out.where(loss != 0, 100)
    return out.where(gain != 0, 0)


def macd(values, fast: int = 12, slow: int = 26, signal: int = 9):
    s = pd.Series(values, dtype="float64")
    line = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def _num(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _normalize(candles: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for c in candles:
        if not isinstance(c, dict):
            continue
        o, h, l, close = (_num(c.get(k)) for k in ("open", "high", "low", "close"))
        vol = _num(c.get("volume", c.get("vol", 0)))
        if min(o, h, l, close) <= 0 or h < max(o, l, close) or l > min(o, h, close):
            continue
        rows.append({"open": o, "high": h, "low": l, "close": close, "volume": max(0, vol)})
    return pd.DataFrame(rows)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    prev = df.close.shift(1)
    tr = pd.concat([
        df.high - df.low,
        (df.high - prev).abs(),
        (df.low - prev).abs(),
    ], axis=1).max(axis=1)
    value = tr.rolling(period).mean().iloc[-1]
    return _num(value, _num((df.high - df.low).tail(period).mean()))


def _flow(df: pd.DataFrame, window: int = 12) -> tuple[float, str]:
    """Pressão assinada do candle, ponderada pelo volume, escala -1..+1."""
    sample = df.tail(window)
    spread = (sample.high - sample.low).replace(0, float("nan"))
    pressure = ((sample.close - sample.open) / spread).clip(-1, 1).fillna(0)
    volume = sample.volume
    if float(volume.sum()) > 0:
        value = float((pressure * volume).sum() / volume.sum())
    else:
        value = float(pressure.mean())
    side = "CALL" if value >= 0.08 else "PUT" if value <= -0.08 else "NEUTRO"
    return value, side


def _confirmed_pivots(values: pd.Series, left: int = 2, right: int = 2):
    """Pivôs confirmados: só considera pivô após os candles à direita fecharem."""
    vals = values.tolist()
    pivots = []
    for i in range(left, len(vals) - right):
        window = vals[i-left:i+right+1]
        v = vals[i]
        if v == max(window) and window.count(v) == 1:
            pivots.append((i, v, "high"))
        elif v == min(window) and window.count(v) == 1:
            pivots.append((i, v, "low"))
    return pivots


def _market_structure(df: pd.DataFrame) -> tuple[str, dict]:
    pivots = _confirmed_pivots(df.high, 2, 2) + _confirmed_pivots(df.low, 2, 2)
    highs = [(i, v) for i, v, kind in pivots if kind == "high"]
    lows = [(i, v) for i, v, kind in pivots if kind == "low"]
    highs.sort(); lows.sort()
    # Usa os dois últimos pivôs confirmados de cada tipo, sem usar o pivô ainda em formação.
    if len(highs) < 2 or len(lows) < 2:
        return "NEUTRO", {"higher_high": None, "higher_low": None, "lower_high": None, "lower_low": None}
    hh = highs[-1][1] > highs[-2][1]
    hl = lows[-1][1] > lows[-2][1]
    lh = highs[-1][1] < highs[-2][1]
    ll = lows[-1][1] < lows[-2][1]
    if hh and hl:
        side = "CALL"
    elif lh and ll:
        side = "PUT"
    else:
        side = "NEUTRO"
    return side, {
        "higher_high": bool(hh), "higher_low": bool(hl),
        "lower_high": bool(lh), "lower_low": bool(ll),
        "last_confirmed_high": float(highs[-1][1]),
        "previous_confirmed_high": float(highs[-2][1]),
        "last_confirmed_low": float(lows[-1][1]),
        "previous_confirmed_low": float(lows[-2][1]),
    }


def _directional_core(df: pd.DataFrame):
    close = df.close
    e9 = ema(close, 9)
    e21 = ema(close, 21)
    # Exige inclinação recente além do posicionamento das médias.
    if e9.iloc[-1] > e21.iloc[-1] and e9.iloc[-1] > e9.iloc[-4]:
        trend = "CALL"
    elif e9.iloc[-1] < e21.iloc[-1] and e9.iloc[-1] < e9.iloc[-4]:
        trend = "PUT"
    else:
        trend = "NEUTRO"
    flow_value, flow_side = _flow(df)
    structure, structure_detail = _market_structure(df)
    votes = [trend, flow_side, structure]
    # Núcleo direcional estrito: os três motores precisam concordar.
    direction = votes[0] if votes[0] != "NEUTRO" and votes.count(votes[0]) == 3 else "NEUTRO"
    return direction, trend, flow_side, flow_value, structure, structure_detail, e9, e21


def analyze_candles(candles: Iterable[dict]) -> dict:
    df = _normalize(candles)
    if len(df) < 50:
        return {"ready": False, "direction": "NEUTRO", "confidence": 0,
                "reason": f"Dados insuficientes: {len(df)}/50 candles válidos."}

    direction, trend, flow_side, flow_value, structure, structure_detail, e9, e21 = _directional_core(df)
    close = df.close
    rsi_values = rsi(close, 14)
    macd_line, macd_signal, macd_hist = macd(close)

    # DMI / ADX (Wilder suavizado)
    up = df.high.diff()
    down = -df.low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev = close.shift(1)
    tr = pd.concat([df.high-df.low, (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, float("nan"))
    di_plus_s = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_s
    di_minus_s = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_s
    dx = 100 * (di_plus_s-di_minus_s).abs() / (di_plus_s+di_minus_s).replace(0, float("nan"))
    adx_s = dx.ewm(alpha=1/14, adjust=False).mean()

    # KDJ(9,3,3)
    low9 = df.low.rolling(9).min()
    high9 = df.high.rolling(9).max()
    rsv = ((close-low9)/(high9-low9).replace(0, float("nan"))*100).fillna(50)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3*k-2*d

    rsi_now = _num(rsi_values.iloc[-1], 50)
    macd_now, macd_sig = _num(macd_line.iloc[-1]), _num(macd_signal.iloc[-1])
    dip, dim = _num(di_plus_s.iloc[-1]), _num(di_minus_s.iloc[-1])
    adx = _num(adx_s.iloc[-1])
    k_now, d_now, j_now = _num(k.iloc[-1]), _num(d.iloc[-1]), _num(j.iloc[-1])

    # Confirmações não escolhem a direção: apenas qualificam o núcleo.
    confirmations = 0
    confirmation_details = []
    if direction == "CALL":
        checks = [
            (macd_now > macd_sig, "MACD comprador"),
            (50 < rsi_now < 75, "RSI em faixa compradora"),
            (dip > dim, "DI+ acima de DI-"),
            (k_now > d_now and j_now >= _num(j.iloc[-2]), "KDJ comprador"),
        ]
    elif direction == "PUT":
        checks = [
            (macd_now < macd_sig, "MACD vendedor"),
            (25 < rsi_now < 50, "RSI em faixa vendedora"),
            (dim > dip, "DI- acima de DI+"),
            (k_now < d_now and j_now <= _num(j.iloc[-2]), "KDJ vendedor"),
        ]
    else:
        checks = []
    for passed, label in checks:
        if passed:
            confirmations += 1
            confirmation_details.append(label)

    # Bloqueia direção quando o núcleo está alinhado, mas as confirmações são muito fracas.
    if direction != "NEUTRO" and confirmations < 2:
        final_direction = "NEUTRO"
        reason = "Núcleo alinhado, mas menos de 2 confirmações técnicas; aguardar."
    else:
        final_direction = direction
        reason = "Tendência, fluxo e estrutura alinhados." if direction != "NEUTRO" else "Sem consenso entre tendência, fluxo e estrutura."

    atr = _atr(df)
    flow_strength = abs(flow_value) * 100
    structure_bonus = 1 if structure != "NEUTRO" else 0
    confidence = max(50, min(97, int(58 + flow_strength * 0.18 + confirmations * 5 + structure_bonus * 4 + min(adx, 35) * 0.15))) if final_direction != "NEUTRO" else 0

    return {
        "ready": True,
        "direction": final_direction,
        "confidence": confidence,
        "price": float(close.iloc[-1]),
        "reason": reason,
        "directional_core": {
            "trend": trend,
            "flow": flow_side,
            "flow_value": round(flow_value, 4),
            "structure": structure,
            "votes": {"trend": trend, "flow": flow_side, "structure": structure},
            "structure_detail": structure_detail,
        },
        "confirmations": {"count": confirmations, "required": 2, "details": confirmation_details},
        "indicators": {
            "ema9": _num(e9.iloc[-1]), "ema21": _num(e21.iloc[-1]),
            "rsi": rsi_now, "macd": macd_now, "macd_signal": macd_sig,
            "macd_histogram": _num(macd_hist.iloc[-1]),
            "di_plus": dip, "di_minus": dim, "adx": adx,
            "k": k_now, "d": d_now, "j": j_now, "atr": atr,
        },
        "flow_source": "ESTIMADO_OHLCV",
        "confidence_type": "SCORE_DE_CONFLUENCIA_NAO_PROBABILIDADE",
    }
