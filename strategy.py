"""Purple Signal — direção por estrutura/Price Action, auxiliares e gestão de risco.
O fluxo é uma estimativa OHLCV; não representa agressão bid/ask nem livro real.
A análise de 1 minuto utiliza somente candles cujo fechamento esteja confirmado.
"""
from __future__ import annotations

from typing import Literal
import time

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
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
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
    volume = pd.to_numeric(
        df.get("volume", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0).clip(lower=0)
    weights = volume.tail(window)
    values = pressure.tail(window)
    if float(weights.sum()) <= 0:
        return float(values.mean())
    return float((values * weights).sum() / weights.sum())


def _hidden_structure_confluence(
    df: pd.DataFrame, direction: str, atr_value: float
) -> tuple[int, str]:
    """Confluência privada de pivôs/suporte/resistência. Não retorna níveis ao painel."""
    if len(df) < 12 or not np.isfinite(atr_value) or atr_value <= 0:
        return 0, "NONE"

    highs = df.high.astype(float).to_numpy()
    lows = df.low.astype(float).to_numpy()
    piv_hi, piv_lo = [], []

    # Pivô confirmado: duas velas à esquerda e duas à direita.
    for i in range(max(2, len(df) - 45), len(df) - 2):
        if highs[i] >= max(highs[i - 2:i]) and highs[i] > max(highs[i + 1:i + 3]):
            piv_hi.append(highs[i])
        if lows[i] <= min(lows[i - 2:i]) and lows[i] < min(lows[i + 1:i + 3]):
            piv_lo.append(lows[i])

    price = float(df.close.iloc[-1])
    tolerance = max(atr_value * 0.35, abs(price) * 0.00005)
    near_support = bool(
        piv_lo and min(abs(price - x) for x in piv_lo[-8:]) <= tolerance
    )
    near_resistance = bool(
        piv_hi and min(abs(price - x) for x in piv_hi[-8:]) <= tolerance
    )
    prior_high = max(piv_hi[-8:]) if piv_hi else None
    prior_low = min(piv_lo[-8:]) if piv_lo else None
    last = df.iloc[-1]
    avg_vol = (
        float(pd.to_numeric(df.volume, errors="coerce").fillna(0).tail(20).mean())
        if "volume" in df else 0
    )
    vol = float(last.get("volume", 0) or 0)
    breakout_up = (
        prior_high is not None
        and price > prior_high + tolerance * 0.15
        and (avg_vol <= 0 or vol >= avg_vol * 1.15)
    )
    breakout_down = (
        prior_low is not None
        and price < prior_low - tolerance * 0.15
        and (avg_vol <= 0 or vol >= avg_vol * 1.15)
    )

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


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """ADX clássico suavizado por Wilder, acompanhado de DI+ e DI-."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_sm = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_sm = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = (100 * plus_sm / atr.replace(0, np.nan)).fillna(0)
    minus_di = (100 * minus_sm / atr.replace(0, np.nan)).fillna(0)
    dx = (
        100 * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    ).fillna(0)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0)
    return adx, plus_di, minus_di, atr


def _flow_diagnostics(
    df: pd.DataFrame, atr_value: float, window: int = 20
) -> dict:
    """Diagnósticos estimados por OHLCV; não substituem agressão nem book real."""
    recent = df.tail(window).copy()
    spread = (recent.high - recent.low).replace(0, np.nan)
    body_ratio = (
        (recent.close - recent.open).abs() / spread
    ).clip(0, 1).fillna(0)
    candle_top = recent[["open", "close"]].max(axis=1)
    candle_bottom = recent[["open", "close"]].min(axis=1)
    upper_wick_ratio = ((recent.high - candle_top) / spread).clip(0, 1).fillna(0)
    lower_wick_ratio = ((candle_bottom - recent.low) / spread).clip(0, 1).fillna(0)
    wick_rejection = (lower_wick_ratio - upper_wick_ratio).clip(-1, 1)
    volume = pd.to_numeric(
        recent.get("volume", pd.Series(0, index=recent.index)),
        errors="coerce",
    ).fillna(0).clip(lower=0)

    vol_ref = float(volume.median()) if len(volume) else 0.0
    last_vol = float(volume.iloc[-1]) if len(volume) else 0.0
    last_body = float(body_ratio.iloc[-1]) if len(body_ratio) else 0.0
    volume_ratio = last_vol / vol_ref if vol_ref > 0 else 0.0
    absorption = bool(vol_ref > 0 and volume_ratio >= 1.5 and last_body <= 0.30)
    last_range = (
        float(spread.iloc[-1])
        if len(spread) and pd.notna(spread.iloc[-1]) else 0.0
    )
    displacement_atr = last_range / atr_value if atr_value > 0 else 0.0
    impact = (
        (last_range / atr_value) / max(volume_ratio, 0.25)
        if atr_value > 0 else 0.0
    )

    return {
        "absorption_detected": absorption,
        "absorption_strength": round(
            min(100.0, volume_ratio * (1 - last_body) * 35), 1
        ),
        "volume_ratio": round(volume_ratio, 2),
        "candle_body_ratio": round(last_body, 3),
        "candle_upper_wick_ratio": round(float(upper_wick_ratio.iloc[-1]), 3) if len(upper_wick_ratio) else 0.0,
        "candle_lower_wick_ratio": round(float(lower_wick_ratio.iloc[-1]), 3) if len(lower_wick_ratio) else 0.0,
        "candle_wick_rejection": round(float(wick_rejection.iloc[-1]), 3) if len(wick_rejection) else 0.0,
        "price_impact_estimate": round(impact, 3),
        "displacement_atr": round(displacement_atr, 3),
    }


def _flow_components(df: pd.DataFrame, window: int = 12) -> dict:
    """Proxies auxiliares OHLCV; nunca decidem a direção principal."""
    recent = df.tail(window).copy()
    spread = (recent.high - recent.low).replace(0, np.nan)
    body_pressure = (
        (recent.close - recent.open) / spread
    ).clip(-1, 1).fillna(0.0)

    # Sombras medem rejeição: inferior favorece pressão compradora; superior,
    # pressão vendedora. O peso é limitado para uma sombra isolada não dominar
    # o fluxo nem transformar automaticamente qualquer pavio em reversão.
    candle_top = recent[["open", "close"]].max(axis=1)
    candle_bottom = recent[["open", "close"]].min(axis=1)
    upper_wick = ((recent.high - candle_top) / spread).clip(0, 1).fillna(0.0)
    lower_wick = ((candle_bottom - recent.low) / spread).clip(0, 1).fillna(0.0)
    wick_pressure = (lower_wick - upper_wick).clip(-1, 1)
    candle_pressure = (0.70 * body_pressure + 0.30 * wick_pressure).clip(-1, 1)

    volume = pd.to_numeric(
        recent.get("volume", pd.Series(0.0, index=recent.index)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    # 1) Fluxo do candle: corpo (70%) + rejeição das sombras (30%).
    # Mantém também o corpo puro disponível para diagnóstico e comparação.
    candle_body_flow = float(body_pressure.mean()) if len(body_pressure) else 0.0
    candle_wick_flow = float(wick_pressure.mean()) if len(wick_pressure) else 0.0
    candle_flow = float(candle_pressure.mean()) if len(candle_pressure) else 0.0

    # 2) Pressão combinada de corpo/sombras ponderada pelo volume disponível.
    volume_available = float(volume.sum()) > 0
    if volume_available:
        candle_volume_flow = float(
            (candle_pressure * volume).sum() / volume.sum()
        )
    else:
        candle_volume_flow = candle_flow

    # 3) Deslocamento líquido normalizado pelo caminho percorrido.
    close_change = recent.close.diff().fillna(0.0)
    path = float(close_change.abs().sum())
    market_flow = float(close_change.sum() / path) if path > 0 else 0.0

    # 4) Desequilíbrio estimado entre volume de barras de alta e de baixa.
    bar_direction = np.sign(close_change)
    signed_volume = float((bar_direction * volume).sum())
    market_volume_flow = (
        signed_volume / float(volume.sum())
        if volume_available else market_flow
    )

    values = {
        "market_flow": float(np.clip(market_flow, -1.0, 1.0)),
        "candle_flow": float(np.clip(candle_flow, -1.0, 1.0)),
        "candle_body_flow": float(np.clip(candle_body_flow, -1.0, 1.0)),
        "candle_wick_flow": float(np.clip(candle_wick_flow, -1.0, 1.0)),
        "candle_volume_flow": float(np.clip(candle_volume_flow, -1.0, 1.0)),
        "market_volume_flow": float(np.clip(market_volume_flow, -1.0, 1.0)),
    }

    # Sem volume, só existem duas leituras independentes.
    active_values = (
        [
            values["market_flow"],
            values["candle_flow"],
            values["candle_volume_flow"],
            values["market_volume_flow"],
        ]
        if volume_available
        else [values["market_flow"], values["candle_flow"]]
    )

    directional_votes = [
        "CALL" if value >= 0.08 else
        "PUT" if value <= -0.08 else
        "NEUTRO"
        for value in active_values
    ]
    bullish = directional_votes.count("CALL")
    bearish = directional_votes.count("PUT")
    majority = len(directional_votes) // 2 + 1

    direction = (
        "CALL" if bullish >= majority else
        "PUT" if bearish >= majority else
        "NEUTRO"
    )
    composite = float(np.mean(active_values)) if active_values else 0.0
    strength = min(100, int(abs(composite) * 100))

    return {
        **values,
        "direction": direction,
        "directional_votes": directional_votes,
        "active_vote_count": len(directional_votes),
        "volume_available": volume_available,
        "pressure": composite,
        "strength": strength,
        "bullish_votes": bullish,
        "bearish_votes": bearish,
    }


def _trend_component(
    close: pd.Series, atr_value: float, window: int = 20
) -> dict:
    """Tendência independente das EMAs 9/21, via inclinação linear do preço."""
    values = close.tail(window).astype(float).to_numpy()
    if len(values) < window or not np.isfinite(atr_value) or atr_value <= 0:
        return {"direction": "NEUTRO", "slope_atr": 0.0, "strength": 0}

    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values, 1)[0])
    normalized = slope / atr_value
    direction = (
        "CALL" if normalized >= 0.015 else
        "PUT" if normalized <= -0.015 else
        "NEUTRO"
    )
    return {
        "direction": direction,
        "slope_atr": normalized,
        "strength": min(100, int(abs(normalized) * 100)),
    }


def _three_candle_morphology(df: pd.DataFrame) -> dict:
    """Lê em conjunto os três candles mais recentes usando somente OHLCV.

    O resultado é uma estimativa morfológica; não representa agressão real,
    negócios no bid/ask ou profundidade do livro de ofertas.
    """
    if len(df) < 3:
        return {
            "direction": "NEUTRO", "valid": False, "reason": "THREE_CANDLES_INSUFFICIENT",
            "score": 0.0, "bullish_count": 0, "bearish_count": 0,
            "candles": [], "sequence": "INDEFINIDA",
        }

    recent = df.tail(3)
    details = []
    scores = []
    directions = []
    volumes = pd.to_numeric(
        recent.get("volume", pd.Series(0.0, index=recent.index)), errors="coerce"
    ).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)

    for (_, row), volume in zip(recent.iterrows(), volumes):
        o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))
        candle_range = h - l
        if (not np.isfinite([o, h, l, c]).all() or candle_range <= 0
                or h < max(o, c) or l > min(o, c)):
            return {
                "direction": "NEUTRO", "valid": False, "reason": "THREE_CANDLES_INVALID",
                "score": 0.0, "bullish_count": 0, "bearish_count": 0,
                "candles": [], "sequence": "INDEFINIDA",
            }

        body = abs(c - o) / candle_range
        upper = (h - max(o, c)) / candle_range
        lower = (min(o, c) - l) / candle_range
        close_position = (c - l) / candle_range
        signed_body = (c - o) / candle_range
        wick_rejection = lower - upper
        close_pressure = (2.0 * close_position) - 1.0

        # Corpo, rejeição dos pavios e posição do fechamento são dimensões
        # distintas. A cor do candle isoladamente não decide a direção.
        score = float(np.clip(
            0.50 * signed_body + 0.30 * wick_rejection + 0.20 * close_pressure,
            -1.0, 1.0,
        ))
        bullish_rejection = lower >= 0.35 and close_position >= 0.62
        bearish_rejection = upper >= 0.35 and close_position <= 0.38
        if bullish_rejection:
            score = max(score, 0.20)
        if bearish_rejection:
            score = min(score, -0.20)

        candle_direction = (
            "CALL" if score >= 0.16 else
            "PUT" if score <= -0.16 else
            "NEUTRO"
        )
        scores.append(score)
        directions.append(candle_direction)
        details.append({
            "body_ratio": round(body, 3),
            "upper_wick_ratio": round(upper, 3),
            "lower_wick_ratio": round(lower, 3),
            "close_position": round(close_position, 3),
            "wick_rejection": round(wick_rejection, 3),
            "pressure_score": round(score, 3),
            "direction": candle_direction,
            "bullish_rejection": bool(bullish_rejection),
            "bearish_rejection": bool(bearish_rejection),
            "volume": float(volume),
        })

    # Dá maior peso ao candle mais recente, sem ignorar os dois anteriores.
    weights = np.array([0.25, 0.33, 0.42], dtype=float)
    score = float(np.dot(np.asarray(scores, dtype=float), weights))
    bullish_count = directions.count("CALL")
    bearish_count = directions.count("PUT")
    strong_opposition = (
        (score > 0 and min(scores) <= -0.60)
        or (score < 0 and max(scores) >= 0.60)
    )
    indecisive = sum(d == "NEUTRO" for d in directions) >= 2

    direction = "NEUTRO"
    if not strong_opposition and not indecisive:
        if bullish_count >= 2 and score >= 0.18:
            direction = "CALL"
        elif bearish_count >= 2 and score <= -0.18:
            direction = "PUT"

    if directions[0] == directions[1] == directions[2] == "CALL":
        sequence = "PRESSAO_COMPRADORA_CONTINUA"
    elif directions[0] == directions[1] == directions[2] == "PUT":
        sequence = "PRESSAO_VENDEDORA_CONTINUA"
    elif bullish_count >= 2:
        sequence = "PREDOMINIO_COMPRADOR_COM_VARIACAO"
    elif bearish_count >= 2:
        sequence = "PREDOMINIO_VENDEDOR_COM_VARIACAO"
    else:
        sequence = "DIVERGENCIA_OU_INDECISAO"

    reason = (
        "CONFIRMED" if direction != "NEUTRO" else
        "STRONG_OPPOSITION" if strong_opposition else
        "INDECISION" if indecisive else
        "DIRECTION_NOT_CONFIRMED"
    )
    return {
        "direction": direction,
        "valid": True,
        "reason": reason,
        "score": score,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "candles": details,
        "sequence": sequence,
    }


def _market_structure_analysis(df: pd.DataFrame) -> dict:
    """Classifica pivôs confirmados, HH/HL, LH/LL e BOS sem vetar entradas."""
    if len(df) < 12:
        return {"label":"INDEFINIDA", "direction":"NEUTRO", "bos":"NONE", "score":0}
    highs=df.high.astype(float).to_numpy(); lows=df.low.astype(float).to_numpy()
    pivot_highs=[]; pivot_lows=[]
    # Dois candles à esquerda e dois à direita: o pivô só existe depois de confirmado.
    for i in range(2, len(df)-2):
        if highs[i] >= max(highs[i-2:i]) and highs[i] > max(highs[i+1:i+3]):
            pivot_highs.append((i, float(highs[i])))
        if lows[i] <= min(lows[i-2:i]) and lows[i] < min(lows[i+1:i+3]):
            pivot_lows.append((i, float(lows[i])))
    hh = len(pivot_highs)>=2 and pivot_highs[-1][1] > pivot_highs[-2][1]
    lh = len(pivot_highs)>=2 and pivot_highs[-1][1] < pivot_highs[-2][1]
    hl = len(pivot_lows)>=2 and pivot_lows[-1][1] > pivot_lows[-2][1]
    ll = len(pivot_lows)>=2 and pivot_lows[-1][1] < pivot_lows[-2][1]
    price=float(df.close.iloc[-1])
    bos_up=bool(pivot_highs and price > pivot_highs[-1][1])
    bos_down=bool(pivot_lows and price < pivot_lows[-1][1])
    if bos_up:
        label, direction, bos = "BOS_ALTA", "CALL", "BOS_UP"
    elif bos_down:
        label, direction, bos = "BOS_BAIXA", "PUT", "BOS_DOWN"
    elif hh and hl:
        label, direction, bos = "HH_HL", "CALL", "NONE"
    elif lh and ll:
        label, direction, bos = "LH_LL", "PUT", "NONE"
    elif hh or hl:
        label, direction, bos = "ESTRUTURA_COMPRADORA_PARCIAL", "CALL", "NONE"
    elif lh or ll:
        label, direction, bos = "ESTRUTURA_VENDEDORA_PARCIAL", "PUT", "NONE"
    else:
        label, direction, bos = "INDEFINIDA", "NEUTRO", "NONE"
    return {
        "label":label, "direction":direction, "bos":bos,
        "higher_high":bool(hh), "higher_low":bool(hl),
        "lower_high":bool(lh), "lower_low":bool(ll),
        "score":5 if direction in ("CALL","PUT") else 0,
    }


def _pullback_analysis(
    df: pd.DataFrame,
    trend_direction: str,
    ema9_series: pd.Series,
    ema21_series: pd.Series,
    atr_value: float,
) -> dict:
    """Confirma retomada da tendência após toque corporal na EMA9 ou EMA21.

    Usa somente os três candles fechados mais recentes. O toque precisa ocorrer
    no corpo de um dos dois candles anteriores; o último candle deve confirmar
    a retomada na direção da tendência. Sombras não contam como reteste.
    """
    empty={"direction":"NEUTRO", "detected":False, "strength":0,
           "ema9_retest":False, "ema21_retest":False,
           "fibonacci_retest":False, "reversal_confirmed":False,
           "reason":"NO_TREND_OR_DATA"}
    if (
        trend_direction not in ("CALL", "PUT")
        or len(df) < 3
        or len(ema9_series) < 3
        or len(ema21_series) < 3
        or atr_value <= 0
    ):
        return empty

    recent = df.tail(3).reset_index(drop=True)
    ema9_recent = ema9_series.tail(3).reset_index(drop=True)
    ema21_recent = ema21_series.tail(3).reset_index(drop=True)

    # A tendência precisa estar organizada e inclinada na mesma direção.
    ema9_now = float(ema9_recent.iloc[-1])
    ema21_now = float(ema21_recent.iloc[-1])
    ema9_previous = float(ema9_recent.iloc[-2])
    ema21_previous = float(ema21_recent.iloc[-2])
    if trend_direction == "CALL":
        trend_aligned = bool(
            ema9_now > ema21_now
            and ema9_now > ema9_previous
            and ema21_now > ema21_previous
        )
    else:
        trend_aligned = bool(
            ema9_now < ema21_now
            and ema9_now < ema9_previous
            and ema21_now < ema21_previous
        )
    if not trend_aligned:
        return {**empty, "reason":"EMA_TREND_NOT_ALIGNED"}

    # O reteste é válido somente quando o corpo (open/close) toca a média.
    # As sombras (high/low) são deliberadamente ignoradas para evitar falsos toques.
    ema9_retest = False
    ema21_retest = False
    touched_levels = []
    for idx in (0, 1):
        candle = recent.iloc[idx]
        body_low = min(float(candle.open), float(candle.close))
        body_high = max(float(candle.open), float(candle.close))
        ema9_level = float(ema9_recent.iloc[idx])
        ema21_level = float(ema21_recent.iloc[idx])

        if body_low <= ema9_level <= body_high:
            ema9_retest = True
            touched_levels.append((idx, "EMA9", ema9_level))
        if body_low <= ema21_level <= body_high:
            ema21_retest = True
            touched_levels.append((idx, "EMA21", ema21_level))

    # O candle mais recente precisa vir DEPOIS do toque e fechar retomando
    # a tendência: corpo direcional + fechamento além da média tocada e
    # além do fechamento do candle imediatamente anterior.
    last = recent.iloc[-1]
    previous = recent.iloc[-2]
    last_open = float(last.open)
    last_close = float(last.close)
    previous_close = float(previous.close)
    reversal = False
    confirmed_level = None

    if touched_levels:
        if trend_direction == "CALL":
            directional_body = last_close > last_open
            for _, level_name, level_value in touched_levels:
                if (
                    directional_body
                    and last_close > level_value
                    and last_close > previous_close
                ):
                    reversal = True
                    confirmed_level = level_name
                    break
        else:
            directional_body = last_close < last_open
            for _, level_name, level_value in touched_levels:
                if (
                    directional_body
                    and last_close < level_value
                    and last_close < previous_close
                ):
                    reversal = True
                    confirmed_level = level_name
                    break

    detected = bool(touched_levels and reversal)
    strength = min(
        100,
        60 + (15 if ema9_retest else 0) + (20 if ema21_retest else 0),
    ) if detected else 0
    return {
        "direction":trend_direction if detected else "NEUTRO",
        "detected":detected,
        "strength":int(strength),
        "ema9_retest":bool(ema9_retest),
        "ema21_retest":bool(ema21_retest),
        "fibonacci_retest":False,
        "reversal_confirmed":bool(reversal),
        "confirmed_level":confirmed_level,
        "reason":"CONFIRMED" if detected else (
            "RETEST_OR_RETURN_MISSING" if trend_aligned else "EMA_TREND_NOT_ALIGNED"
        ),
    }


def _closed_one_minute_candles(candles: list[dict], now: float | None = None) -> tuple[list[dict], dict]:
    """Retém candles fechados usando a cadência temporal observada (M1/M5/M15)."""
    if not isinstance(candles, (list, tuple)) or not candles:
        return [], {"ok": False, "reason": "SEM_CANDLES"}
    now_ts = float(time.time() if now is None else now)
    normalized = []
    for candle in candles:
        if not isinstance(candle, dict):
            continue
        raw_ts = candle.get("ts", candle.get("from"))
        try:
            ts = float(raw_ts)
        except (TypeError, ValueError):
            return [], {"ok": False, "reason": "TIMESTAMP_AUSENTE_OU_INVALIDO"}
        if not np.isfinite(ts) or ts <= 0:
            return [], {"ok": False, "reason": "TIMESTAMP_AUSENTE_OU_INVALIDO"}
        if ts > 1_000_000_000_000:  # milissegundos -> segundos
            ts /= 1000.0
        item = dict(candle)
        item["_candle_start_ts"] = ts
        normalized.append(item)
    if not normalized:
        return [], {"ok": False, "reason": "SEM_CANDLES_VALIDOS"}
    normalized.sort(key=lambda item: item["_candle_start_ts"])
    unique = {item["_candle_start_ts"]: item for item in normalized}
    ordered = [unique[key] for key in sorted(unique)]
    starts = np.asarray([item["_candle_start_ts"] for item in ordered], dtype=float)
    deltas = np.diff(starts)
    deltas = deltas[deltas > 0]
    observed_interval = float(np.median(deltas[-30:])) if len(deltas) else 60.0
    supported_intervals = (60, 300, 900)
    timeframe_seconds = min(supported_intervals, key=lambda value: abs(value - observed_interval))
    if abs(timeframe_seconds - observed_interval) > max(3.0, timeframe_seconds * 0.10):
        return [], {"ok": False, "reason": "INTERVALO_DOS_CANDLES_NAO_RECONHECIDO",
                    "observed_interval_seconds": observed_interval}
    closed = [item for item in ordered if item["_candle_start_ts"] + timeframe_seconds <= now_ts]
    if not closed:
        return [], {
            "ok": False, "reason": "AGUARDANDO_FECHAMENTO_DO_CANDLE_1M",
            "closed_count": 0,
            "timeframe_seconds": timeframe_seconds,
            "latest_candle_start": ordered[-1]["_candle_start_ts"],
            "next_close_at": ordered[-1]["_candle_start_ts"] + timeframe_seconds,
        }
    latest_start = ordered[-1]["_candle_start_ts"]
    latest_open_excluded = latest_start + timeframe_seconds > now_ts
    for item in closed:
        item.pop("_candle_start_ts", None)
    return closed, {
        "ok": True, "timeframe_seconds": timeframe_seconds, "closed_count": len(closed),
        "latest_candle_start": latest_start,
        "latest_was_open_and_excluded": bool(latest_open_excluded),
        "last_closed_at": float(closed[-1].get("ts", closed[-1].get("from"))) + timeframe_seconds,
    }


def _divergence_analysis(
    df: pd.DataFrame, rsi_series: pd.Series, macd_hist: pd.Series, atr_value: float
) -> dict:
    """Detecta divergências clássicas em pivôs confirmados; nunca veta sozinha."""
    result = {"direction": "NEUTRO", "label": "NONE", "bullish": False,
              "bearish": False, "rsi_confirmed": False, "macd_confirmed": False,
              "strength": 0}
    if len(df) < 20 or len(rsi_series) != len(df) or len(macd_hist) != len(df):
        return result
    highs = df.high.astype(float).to_numpy()
    lows = df.low.astype(float).to_numpy()
    rsi_values = rsi_series.astype(float).to_numpy()
    macd_values = macd_hist.astype(float).to_numpy()
    pivot_highs, pivot_lows = [], []
    for idx in range(max(2, len(df) - 60), len(df) - 2):
        if highs[idx] >= max(highs[idx-2:idx]) and highs[idx] > max(highs[idx+1:idx+3]):
            pivot_highs.append(idx)
        if lows[idx] <= min(lows[idx-2:idx]) and lows[idx] < min(lows[idx+1:idx+3]):
            pivot_lows.append(idx)
    tolerance = max(float(atr_value) * 0.05, abs(float(df.close.iloc[-1])) * 0.00001)
    if len(pivot_lows) >= 2:
        a, b = pivot_lows[-2:]
        lower_low = lows[b] < lows[a] - tolerance
        rsi_higher = rsi_values[b] > rsi_values[a] + 1.0
        macd_higher = macd_values[b] > macd_values[a]
        if lower_low and (rsi_higher or macd_higher):
            result.update({"direction": "CALL", "label": "BULLISH_DIVERGENCE",
                           "bullish": True, "rsi_confirmed": bool(rsi_higher),
                           "macd_confirmed": bool(macd_higher),
                           "strength": 2 if rsi_higher and macd_higher else 1})
    if len(pivot_highs) >= 2:
        a, b = pivot_highs[-2:]
        higher_high = highs[b] > highs[a] + tolerance
        rsi_lower = rsi_values[b] < rsi_values[a] - 1.0
        macd_lower = macd_values[b] < macd_values[a]
        if higher_high and (rsi_lower or macd_lower):
            if result["bullish"]:
                result.update({"direction": "NEUTRO", "label": "CONFLICTING_DIVERGENCES", "strength": 0})
            else:
                result.update({"direction": "PUT", "label": "BEARISH_DIVERGENCE",
                               "bearish": True, "rsi_confirmed": bool(rsi_lower),
                               "macd_confirmed": bool(macd_lower),
                               "strength": 2 if rsi_lower and macd_lower else 1})
    return result

Direction = Literal["CALL", "PUT", "NEUTRO"]


def analyze_candles(candles: list[dict], brain_profile: dict | None = None) -> dict:
    """Analisa candles fechados; brain_profile é aceito para compatibilidade com server_v3."""
    closed_candles, candle_clock = _closed_one_minute_candles(candles)
    if not candle_clock.get("ok"):
        return {"ready": False, "reason": candle_clock.get("reason", "FECHAMENTO_1M_NAO_CONFIRMADO"),
                "candle_clock": candle_clock}
    # EMA100 exige aquecimento técnico; não é um filtro de entrada.
    if len(closed_candles) < 100:
        return {"ready": False,
                "reason": "Candles fechados insuficientes para EMA100 (mínimo técnico: 100)",
                "candle_clock": candle_clock}

    df = pd.DataFrame(closed_candles).copy()
    for col in ("open", "high", "low", "close"):
        if col not in df:
            return {"ready": False, "reason": f"Campo obrigatório ausente: {col}"}
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df:
        df["volume"] = 0.0
    df["volume"] = pd.to_numeric(
        df["volume"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)

    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    valid = (
        (df.high >= df[["open", "close", "low"]].max(axis=1))
        & (df.low <= df[["open", "close", "high"]].min(axis=1))
    )
    df = df.loc[valid].reset_index(drop=True)

    if len(df) < 100:
        return {"ready": False,
                "reason": "Candles fechados válidos insuficientes para EMA100 (mínimo técnico: 100)",
                "candle_clock": candle_clock}

    close, high, low = df.close, df.high, df.low
    ma9, ma21, ma100 = ema(close, 9), ema(close, 21), ema(close, 100)
    macd_line, macd_signal, macd_hist = macd(close)
    rsi14 = rsi(close, 14)
    k, d, j = kdj(high, low, close, 9)
    adx_series, plus_di, minus_di, atr = _adx(high, low, close, 14)

    i, ip = len(df) - 1, len(df) - 2
    price = float(close.iloc[i])
    atr_value = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
    ma9_value = float(ma9.iloc[i])
    ma21_value = float(ma21.iloc[i])
    ma100_value = float(ma100.iloc[i])
    adx_value = float(adx_series.iloc[i])
    dp, dm = float(plus_di.iloc[i]), float(minus_di.iloc[i])
    rsi_value = float(rsi14.iloc[i])

    # CAMADA 1 — direção principal: estrutura + Price Action + tendência de preço.
    # Fluxo, médias e osciladores são auxiliares e não decidem a direção.
    flow_data = _flow_components(df)
    trend_data = _trend_component(close, atr_value)
    morphology_data = _three_candle_morphology(df)
    morphology_direction = morphology_data["direction"]
    flow_direction = flow_data["direction"]
    trend_direction = trend_data["direction"]
    flow = flow_data["pressure"]
    flow_strength = flow_data["strength"]
    structure_data = _market_structure_analysis(df)
    divergence_data = _divergence_analysis(df, rsi14, macd_hist, atr_value)

    primary_components = [
        (structure_data["direction"], 0.45),
        (morphology_direction, 0.30),
        (trend_direction, 0.25),
    ]
    primary_score = sum(
        weight * (1 if component_direction == "CALL" else -1 if component_direction == "PUT" else 0)
        for component_direction, weight in primary_components
    )
    # Divergência relevante reforça ou reduz a leitura, mas nunca a veta sozinha.
    if divergence_data["direction"] in ("CALL", "PUT"):
        primary_score += 0.10 if divergence_data["direction"] == "CALL" else -0.10
    primary_score = float(np.clip(primary_score, -1.0, 1.0))
    direction: Direction = (
        "CALL" if primary_score > 0 else
        "PUT" if primary_score < 0 else
        "NEUTRO"
    )
    direction_votes = [component_direction for component_direction, _ in primary_components]
    direction_bullish_votes = direction_votes.count("CALL")
    direction_bearish_votes = direction_votes.count("PUT")
    direction_strength = min(100, int(abs(primary_score) * 100))

    # CAMADA 2 — EMAs são confluências auxiliares, nunca pré-requisito direcional.
    ma9_previous = float(ma9.iloc[i - 3])
    ma21_previous = float(ma21.iloc[i - 3])
    ma_bull = (
        ma9_value > ma21_value
        and ma9_value > ma9_previous
        and ma21_value > ma21_previous
    )
    ma_bear = (
        ma9_value < ma21_value
        and ma9_value < ma9_previous
        and ma21_value < ma21_previous
    )
    ema_direction = "CALL" if ma_bull else ("PUT" if ma_bear else "NEUTRO")

    # EMA100 é referência auxiliar de contexto, não filtro obrigatório.
    ema100_previous = float(ma100.iloc[i - 3])
    ema100_direction = (
        "CALL" if price > ma100_value and ma100_value > ema100_previous else
        "PUT" if price < ma100_value and ma100_value < ema100_previous else
        "NEUTRO"
    )
    ema100_bull = ema100_direction == "CALL"
    ema100_bear = ema100_direction == "PUT"

    ema_spread_atr = (
        abs(ma9_value - ma21_value) / atr_value if atr_value > 0 else 0.0
    )
    ema_slope_atr = (
        abs(ma9_value - ma9_previous) / atr_value if atr_value > 0 else 0.0
    )
    sideways = bool(adx_value < 16)

    # Filtro adaptativo de volatilidade.
    atr_reference = atr.tail(50).dropna()
    atr_median = float(atr_reference.median()) if len(atr_reference) else 0.0
    atr_ratio = atr_value / atr_median if atr_median > 0 else 1.0
    volatility_too_low = bool(atr_median > 0 and atr_ratio < 0.45)
    volatility_spike = bool(atr_median > 0 and atr_ratio > 2.50)

    trend_strength = (
        "FORTE" if adx_value >= 25 else
        "MODERADA" if adx_value >= 20 else
        "FRACA" if adx_value >= 16 else
        "LATERAL"
    )

    # Confirmações direcionais.
    macd_bull = (
        float(macd_line.iloc[i]) > float(macd_signal.iloc[i])
        and float(macd_hist.iloc[i]) >= float(macd_hist.iloc[ip])
    )
    macd_bear = (
        float(macd_line.iloc[i]) < float(macd_signal.iloc[i])
        and float(macd_hist.iloc[i]) <= float(macd_hist.iloc[ip])
    )
    rsi_bull = 50 < rsi_value < 80
    rsi_bear = 20 < rsi_value < 50
    di_bull, di_bear = dp > dm, dm > dp
    adx_confirms = adx_value >= 16

    k_now, d_now, j_now = float(k.iloc[i]), float(d.iloc[i]), float(j.iloc[i])
    kdj_bull = k_now > d_now and j_now > float(j.iloc[ip])
    kdj_bear = k_now < d_now and j_now < float(j.iloc[ip])
    kdj_direction = "CALL" if kdj_bull else ("PUT" if kdj_bear else "NEUTRO")

    confirms_for = {
        "CALL": [
            macd_bull, rsi_bull, di_bull,
            kdj_bull, ma_bull, ema100_bull,
        ],
        "PUT": [
            macd_bear, rsi_bear, di_bear,
            kdj_bear, ma_bear, ema100_bear,
        ],
    }
    confirm_count = (
        sum(bool(x) for x in confirms_for[direction])
        if direction != "NEUTRO" else 0
    )

    # Núcleo 2: retorno à média pela EMA 21.
    distance_from_ema21_atr = (
        (price - ma21_value) / atr_value if atr_value > 0 else 0.0
    )
    previous_price = float(close.iloc[ip])
    last_open = float(df.open.iloc[i])
    last_close = float(df.close.iloc[i])
    last_high = float(df.high.iloc[i])
    last_low = float(df.low.iloc[i])
    candle_range = max(last_high - last_low, 1e-12)
    lower_wick = min(last_open, last_close) - last_low
    upper_wick = last_high - max(last_open, last_close)

    bullish_reversal_candle = (
        last_close > last_open
        and (
            lower_wick / candle_range >= 0.25
            or last_close > previous_price
        )
    )
    bearish_reversal_candle = (
        last_close < last_open
        and (
            upper_wick / candle_range >= 0.25
            or last_close < previous_price
        )
    )

    # Fibonacci: zonas de confluência, não gatilhos isolados.
    fib_window = min(100, len(df))
    fib_high = float(high.tail(fib_window).max())
    fib_low = float(low.tail(fib_window).min())
    fib_range = fib_high - fib_low
    fib_levels = {}
    fib_zone = "NONE"
    fib_position = 0.0
    fib_confirmation = False
    fib_direction = "NEUTRO"
    fib_tolerance = max(atr_value * 0.20, abs(price) * 0.00005)

    if fib_range > 0:
        fib_up = {
            "38.2": fib_high - fib_range * 0.382,
            "50.0": fib_high - fib_range * 0.500,
            "61.8": fib_high - fib_range * 0.618,
        }
        fib_down = {
            "38.2": fib_low + fib_range * 0.382,
            "50.0": fib_low + fib_range * 0.500,
            "61.8": fib_low + fib_range * 0.618,
        }
        fib_levels = fib_up if trend_direction == "CALL" else fib_down
        nearest_fib_name, nearest_fib_price = min(
            fib_levels.items(), key=lambda item: abs(price - item[1])
        )
        fib_position = (price - fib_low) / fib_range

        if abs(price - nearest_fib_price) <= fib_tolerance:
            fib_zone = nearest_fib_name
            fib_confirmation = (
                (trend_direction == "CALL" and bullish_reversal_candle)
                or (trend_direction == "PUT" and bearish_reversal_candle)
            )
            if fib_confirmation:
                fib_direction = trend_direction
                if direction == fib_direction:
                    confirm_count += 1

    mean_reversion_direction = "NEUTRO"
    mean_reversion_confirmations = 0

    if (
        atr_value > 0
        and distance_from_ema21_atr <= -1.0
        and bullish_reversal_candle
    ):
        mean_reversion_confirmations = sum(
            [macd_bull, rsi_value < 50, di_bull, kdj_bull]
        )
        if mean_reversion_confirmations >= 2:
            mean_reversion_direction = "CALL"

    elif (
        atr_value > 0
        and distance_from_ema21_atr >= 1.0
        and bearish_reversal_candle
    ):
        mean_reversion_confirmations = sum(
            [macd_bear, rsi_value > 50, di_bear, kdj_bear]
        )
        if mean_reversion_confirmations >= 2:
            mean_reversion_direction = "PUT"

    signal_mode = "STRUCTURE_PRICE_ACTION"
    if mean_reversion_direction != "NEUTRO":
        # Reversão pode contrariar a tendência principal, mas não um fluxo
        # estimado fortemente oposto. Fluxo OHLCV fraco/neutro não veta o setup.
        reversal_sign = 1.0 if mean_reversion_direction == "CALL" else -1.0
        strong_opposing_flow = flow * reversal_sign <= -0.35
        divergence_supports_reversal = divergence_data["direction"] == mean_reversion_direction
        price_action_supports_reversal = morphology_direction == mean_reversion_direction
        structure_opposes_reversal = (
            structure_data["direction"] in ("CALL", "PUT")
            and structure_data["direction"] != mean_reversion_direction
        )
        if (
            (direction == "NEUTRO" or direction == mean_reversion_direction
             or not structure_opposes_reversal or divergence_supports_reversal
             or price_action_supports_reversal)
            and (not strong_opposing_flow or divergence_supports_reversal or price_action_supports_reversal)
        ):
            direction = mean_reversion_direction
            signal_mode = "MEAN_REVERSION"
            confirm_count = mean_reversion_confirmations

    # Confluências auxiliares pontuam a leitura; não exigimos número fixo de votos.

    diagnostics = _flow_diagnostics(df, atr_value)
    last_candle_direction = (
        "CALL" if last_close > last_open else
        "PUT" if last_close < last_open else
        "NEUTRO"
    )
    absorption_conflict = (
        diagnostics["absorption_detected"]
        and direction != "NEUTRO"
        and last_candle_direction not in ("NEUTRO", direction)
    )

    structure_bonus, _private_structure_tag = _hidden_structure_confluence(
        df, direction, atr_value
    )
    structure_directional_bonus = (
        5 if direction != "NEUTRO" and structure_data["direction"] == direction else
        -5 if direction != "NEUTRO" and structure_data["direction"] in ("CALL", "PUT") else
        0
    )
    pullback_data = _pullback_analysis(
        df, trend_direction, ma9, ma21, atr_value
    )
    brain_score = 0
    entry_filter_reasons = []
    entry_context_warnings = []
    auxiliary_score = 0
    if direction != "NEUTRO":
        direction_sign = 1.0 if direction == "CALL" else -1.0
        # CAMADA 2: indicadores, fluxo e contexto ajustam a qualidade sem exigir unanimidade.
        auxiliary_score += 5 if (ma_bull if direction == "CALL" else ma_bear) else 0
        auxiliary_score += 4 if (ema100_bull if direction == "CALL" else ema100_bear) else 0
        auxiliary_score += 5 if (macd_bull if direction == "CALL" else macd_bear) else 0
        auxiliary_score += 4 if (rsi_bull if direction == "CALL" else rsi_bear) else 0
        auxiliary_score += 3 if (di_bull if direction == "CALL" else di_bear) else 0
        auxiliary_score += 3 if (kdj_bull if direction == "CALL" else kdj_bear) else 0
        if flow * direction_sign > 0:
            auxiliary_score += min(8, flow_strength // 8)
        elif flow * direction_sign < -0.20:
            auxiliary_score -= 5
            entry_context_warnings.append("FLUXO_AUXILIAR_CONTRARIO")
        elif flow_direction != direction:
            entry_context_warnings.append("FLUXO_AUXILIAR_SEM_CONFLUENCIA")

        if morphology_direction == direction:
            auxiliary_score += 4
        elif morphology_direction in ("CALL", "PUT"):
            auxiliary_score -= 3
            entry_context_warnings.append("PRICE_ACTION_CURTO_DIVERGENTE")
        if divergence_data["direction"] == direction:
            auxiliary_score += 7 if divergence_data["strength"] >= 2 else 4
            entry_context_warnings.append("DIVERGENCIA_FAVORAVEL")
        elif divergence_data["direction"] in ("CALL", "PUT"):
            auxiliary_score -= 6
            entry_context_warnings.append("DIVERGENCIA_CONTRARIA_REQUER_ATENCAO")

        auxiliary_score += 6 if adx_value >= 25 else 3 if adx_value >= 20 else 0
        if sideways:
            auxiliary_score -= 3
            entry_context_warnings.append("ADX_EM_FAIXA_LATERAL_CONTEXTO")
        if trend_direction != direction:
            auxiliary_score -= 4
            entry_context_warnings.append("TENDENCIA_DIVERGENTE_AUXILIAR")
        if ema100_direction != direction:
            entry_context_warnings.append("EMA100_DIVERGENTE_AUXILIAR")
        if pullback_data["detected"] and pullback_data["direction"] == direction:
            auxiliary_score += 5
        if fib_confirmation and fib_direction == direction:
            auxiliary_score += 3
        if diagnostics["absorption_detected"]:
            auxiliary_score -= 3
            entry_context_warnings.append("ABSORCAO_ESTIMADA_DETECTADA")
        if absorption_conflict:
            auxiliary_score -= 5
            entry_context_warnings.append("POSSIVEL_CONFLITO_DE_ABSORCAO")

        # A estrutura/Price Action domina a nota; auxiliares apenas ajustam.
        brain_score = int(np.clip(48 + abs(primary_score) * 25 + auxiliary_score, 0, 97))

        # CAMADA 3 — bloqueios operacionais/mercado realmente adversos.
        # Limites financeiros e autorização de execução continuam no servidor.
        if volatility_too_low:
            entry_filter_reasons.append("VOLATILIDADE_BAIXA")
        if volatility_spike:
            entry_filter_reasons.append("PICO_DE_VOLATILIDADE")

    if direction == "NEUTRO":
        entry_filter_reasons.append("DIRECAO_PRIMARIA_INSUFICIENTE")

    if entry_filter_reasons:
        direction = "NEUTRO"
        brain_score = 0

    confidence = int(brain_score) if direction != "NEUTRO" else 0

    macd_cross_up = (
        float(macd_line.iloc[ip]) <= float(macd_signal.iloc[ip])
        and macd_bull
    )
    macd_cross_down = (
        float(macd_line.iloc[ip]) >= float(macd_signal.iloc[ip])
        and macd_bear
    )
    di_cross_up = (
        float(plus_di.iloc[ip]) <= float(minus_di.iloc[ip])
        and di_bull
    )
    di_cross_down = (
        float(plus_di.iloc[ip]) >= float(minus_di.iloc[ip])
        and di_bear
    )

    indicators = {
        "decision_architecture": "ESTRUTURA_PRICE_ACTION_AUXILIARES_RISCO",
        "primary_direction_score": round(primary_score, 3),
        "primary_components": {
            "market_structure": structure_data["direction"],
            "price_action_3_candles": morphology_direction,
            "price_trend": trend_direction,
        },
        "divergence_direction": divergence_data["direction"],
        "divergence_label": divergence_data["label"],
        "divergence_bullish": bool(divergence_data["bullish"]),
        "divergence_bearish": bool(divergence_data["bearish"]),
        "divergence_rsi_confirmed": bool(divergence_data["rsi_confirmed"]),
        "divergence_macd_confirmed": bool(divergence_data["macd_confirmed"]),
        "divergence_strength": int(divergence_data["strength"]),
        "auxiliary_score": int(auxiliary_score),
        "risk_layer_status": "BLOCKED" if entry_filter_reasons else "CLEAR_FOR_SERVER_RISK_CHECK",
        "risk_layer_blockers": list(entry_filter_reasons),
        "candle_timeframe_seconds": int(candle_clock.get("timeframe_seconds", 60)),
        "candle_closed_count": int(candle_clock.get("closed_count", 0)),
        "latest_open_candle_excluded": bool(candle_clock.get("latest_was_open_and_excluded", False)),
        "flow_direction": flow_direction,
        "flow_pressure": round(flow, 3),
        "flow_strength": flow_strength,
        "flow_source": "ESTIMADO_OHLCV",
        "flow_volume_available": bool(flow_data["volume_available"]),
        "flow_active_vote_count": int(flow_data["active_vote_count"]),
        "market_flow": round(flow_data["market_flow"], 3),
        "candle_flow": round(flow_data["candle_flow"], 3),
        "candle_body_flow": round(flow_data["candle_body_flow"], 3),
        "candle_wick_flow": round(flow_data["candle_wick_flow"], 3),
        "candle_volume_flow": round(flow_data["candle_volume_flow"], 3),
        "market_volume_flow": round(flow_data["market_volume_flow"], 3),
        "flow_bullish_votes": int(flow_data["bullish_votes"]),
        "flow_bearish_votes": int(flow_data["bearish_votes"]),
        "three_candle_source": "ESTIMADO_OHLCV",
        "three_candle_direction": morphology_direction,
        "three_candle_valid": bool(morphology_data["valid"]),
        "three_candle_reason": morphology_data["reason"],
        "three_candle_score": round(float(morphology_data["score"]), 3),
        "three_candle_bullish_count": int(morphology_data["bullish_count"]),
        "three_candle_bearish_count": int(morphology_data["bearish_count"]),
        "three_candle_sequence": morphology_data["sequence"],
        "three_candle_details": morphology_data["candles"],
        "pullback_detected": bool(pullback_data["detected"]),
        "pullback_direction": pullback_data["direction"],
        "pullback_strength": int(pullback_data["strength"]),
        "pullback_ema9_retest": bool(pullback_data["ema9_retest"]),
        "pullback_ema21_retest": bool(pullback_data["ema21_retest"]),
        "pullback_fibonacci_retest": bool(pullback_data["fibonacci_retest"]),
        "pullback_reversal_confirmed": bool(pullback_data["reversal_confirmed"]),
        "pullback_reason": pullback_data["reason"],
        "market_structure_label": structure_data["label"],
        "market_structure_direction": structure_data["direction"],
        "market_structure_bos": structure_data["bos"],
        "market_structure_hh": bool(structure_data.get("higher_high", False)),
        "market_structure_hl": bool(structure_data.get("higher_low", False)),
        "market_structure_lh": bool(structure_data.get("lower_high", False)),
        "market_structure_ll": bool(structure_data.get("lower_low", False)),
        "market_structure_score": int(structure_directional_bonus),
        "trend_direction": trend_direction,
        "trend_slope_atr": round(trend_data["slope_atr"], 4),
        "trend_component_strength": int(trend_data["strength"]),
        "direction_trend_vote": trend_direction,
        "direction_bullish_votes": int(direction_bullish_votes),
        "direction_bearish_votes": int(direction_bearish_votes),
        "direction_vote_count": int(len(direction_votes)),
        "direction_strength": int(direction_strength),
        "trend_strength": trend_strength,
        "sideways_market": sideways,
        "atr_value": round(atr_value, 8),
        "atr_median_50": round(atr_median, 8),
        "atr_relative_ratio": round(atr_ratio, 3),
        "volatility_too_low": volatility_too_low,
        "volatility_spike": volatility_spike,
        "entry_filter_passed": not bool(entry_filter_reasons),
        "entry_filter_reasons": entry_filter_reasons,
        "entry_context_warnings": entry_context_warnings,
        "ema_spread_atr": round(ema_spread_atr, 3),
        "ema_slope_atr": round(ema_slope_atr, 3),
        "ema9": round(ma9_value, 6),
        "ema21": round(ma21_value, 6),
        "ema100": round(ma100_value, 6),
        "ema_confirmation": ema_direction,
        "ema100_direction": ema100_direction,
        "ema100_filter": ema100_direction,
        "kdj_k": round(k_now, 2),
        "kdj_d": round(d_now, 2),
        "kdj_j": round(j_now, 2),
        "kdj_direction": kdj_direction,
        "macd": round(float(macd_line.iloc[i]), 6),
        "macd_signal": round(float(macd_signal.iloc[i]), 6),
        "macd_status": (
            "COMPRA" if macd_cross_up else
            "VENDA" if macd_cross_down else
            "ALTA" if macd_bull else
            "BAIXA"
        ),
        "rsi": round(rsi_value, 2),
        "rsi_status": (
            "SOBRECOMPRA" if rsi_value >= 80 else
            "SOBREVENDA" if rsi_value <= 20 else
            "COMPRA" if rsi_bull else
            "VENDA" if rsi_bear else
            "NEUTRO"
        ),
        "rsi_period": 14,
        "rsi_overbought": 80,
        "rsi_oversold": 20,
        "adx": round(adx_value, 2),
        "adx_status": (
            "FORTE" if adx_value >= 25 else
            "MODERADO" if adx_value >= 20 else
            "FRACO"
        ),
        "di_plus": round(dp, 2),
        "di_minus": round(dm, 2),
        "didi_status": (
            "AGULHADA COMPRA" if di_cross_up else
            "AGULHADA VENDA" if di_cross_down else
            "DI+ > DI-" if di_bull else
            "DI- > DI+"
        ),
        "entry_confirmations": int(confirm_count),
        "brain_score": int(brain_score),
        "signal_mode": signal_mode if direction != "NEUTRO" else "NONE",
        "mean_reversion_direction": mean_reversion_direction,
        "mean_reversion_distance_atr": round(distance_from_ema21_atr, 3),
        "mean_reversion_threshold_atr": 1.0,
        "mean_reversion_confirmations": int(mean_reversion_confirmations),
        "fibonacci_zone": fib_zone,
        "fibonacci_direction": fib_direction,
        "fibonacci_confirmation": bool(fib_confirmation),
        "fibonacci_position": round(fib_position, 4),
        "fibonacci_38_2": round(float(fib_levels.get("38.2", 0.0)), 6),
        "fibonacci_50": round(float(fib_levels.get("50.0", 0.0)), 6),
        "fibonacci_61_8": round(float(fib_levels.get("61.8", 0.0)), 6),
        "fibonacci_tolerance": round(fib_tolerance, 6),
        "fibonacci_lookback": int(fib_window),
        "absorption_source": "ESTIMADO_OHLCV",
        **diagnostics,
    }

    return {
        "ready": True,
        "direction": direction,
        "confidence": confidence,
        "price": price,
        "candle_clock": candle_clock,
        "indicators": indicators,
    }
