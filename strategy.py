"""Purple Signal — fluxo, pressão, pullback e estrutura como núcleo direcional.
A leitura de fluxo é uma estimativa baseada em OHLCV (corpo, sombras e volume quando disponível),
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



def _market_structure_analysis(df: pd.DataFrame, atr_value: float) -> dict:
    """Classifica a estrutura confirmada de topos/fundos."""
    empty = {
        "direction": "NEUTRO", "structure": "INDEFINIDA",
        "higher_high": False, "higher_low": False,
        "lower_high": False, "lower_low": False,
        "last_swing_high": 0.0, "last_swing_low": 0.0,
        "breakout": "NONE",
    }
    if len(df) < 12 or not np.isfinite(atr_value) or atr_value <= 0:
        return empty

    highs = df.high.astype(float).to_numpy()
    lows = df.low.astype(float).to_numpy()
    pivot_highs, pivot_lows = [], []
    for idx in range(max(2, len(df) - 80), len(df) - 2):
        if highs[idx] >= max(highs[idx - 2:idx]) and highs[idx] > max(highs[idx + 1:idx + 3]):
            pivot_highs.append((idx, float(highs[idx])))
        if lows[idx] <= min(lows[idx - 2:idx]) and lows[idx] < min(lows[idx + 1:idx + 3]):
            pivot_lows.append((idx, float(lows[idx])))

    tolerance = max(atr_value * 0.08, abs(float(df.close.iloc[-1])) * 0.00001)
    hh = lh = hl = ll = False
    if len(pivot_highs) >= 2:
        hh = pivot_highs[-1][1] > pivot_highs[-2][1] + tolerance
        lh = pivot_highs[-1][1] < pivot_highs[-2][1] - tolerance
    if len(pivot_lows) >= 2:
        hl = pivot_lows[-1][1] > pivot_lows[-2][1] + tolerance
        ll = pivot_lows[-1][1] < pivot_lows[-2][1] - tolerance

    price = float(df.close.iloc[-1])
    last_high = pivot_highs[-1][1] if pivot_highs else 0.0
    last_low = pivot_lows[-1][1] if pivot_lows else 0.0
    breakout = "NONE"
    if last_high and price > last_high + tolerance:
        breakout = "BULLISH_BOS"
    elif last_low and price < last_low - tolerance:
        breakout = "BEARISH_BOS"

    if hh and hl:
        direction, structure = "CALL", "TOPO_E_FUNDO_ASCENDENTES"
    elif lh and ll:
        direction, structure = "PUT", "TOPO_E_FUNDO_DESCENDENTES"
    elif breakout == "BULLISH_BOS":
        direction, structure = "CALL", "QUEBRA_DE_ESTRUTURA_ALTA"
    elif breakout == "BEARISH_BOS":
        direction, structure = "PUT", "QUEBRA_DE_ESTRUTURA_BAIXA"
    elif hh or hl:
        direction, structure = "NEUTRO", "ESTRUTURA_PARCIALMENTE_ALTISTA"
    elif lh or ll:
        direction, structure = "NEUTRO", "ESTRUTURA_PARCIALMENTE_BAIXISTA"
    else:
        direction, structure = "NEUTRO", "INDEFINIDA"

    return {
        "direction": direction, "structure": structure,
        "higher_high": bool(hh), "higher_low": bool(hl),
        "lower_high": bool(lh), "lower_low": bool(ll),
        "last_swing_high": round(last_high, 8),
        "last_swing_low": round(last_low, 8),
        "breakout": breakout,
    }


def _pullback_analysis(
    df: pd.DataFrame, trend_direction: str, ema21_value: float,
    atr_value: float, fib_zone: str,
    bullish_reversal: bool, bearish_reversal: bool,
) -> dict:
    """Detecta retração em zona técnica seguida de retomada da tendência."""
    result = {"detected": False, "direction": "NEUTRO", "zone": "NONE", "strength": 0}
    if len(df) < 5 or atr_value <= 0 or trend_direction not in ("CALL", "PUT"):
        return result

    recent = df.tail(5)
    price = float(df.close.iloc[-1])
    previous_close = float(df.close.iloc[-2])
    tolerance = atr_value * 0.45

    if trend_direction == "CALL":
        extreme = float(recent.low.min())
        ema_retest = abs(extreme - ema21_value) <= tolerance
        reversal = bullish_reversal and price > previous_close
    else:
        extreme = float(recent.high.max())
        ema_retest = abs(extreme - ema21_value) <= tolerance
        reversal = bearish_reversal and price < previous_close

    fib_retest = fib_zone in ("38.2", "50.0", "61.8")
    zone = (
        "EMA21_FIBONACCI" if ema_retest and fib_retest else
        "EMA21" if ema_retest else
        "FIBONACCI" if fib_retest else
        "NONE"
    )
    detected = zone != "NONE" and reversal
    if detected:
        strength = 55 + (20 if ema_retest else 0) + (15 if fib_retest else 0)
        return {
            "detected": True, "direction": trend_direction,
            "zone": zone, "strength": min(100, strength),
        }
    result["zone"] = zone
    return result


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

    last_signed_body = (
        float((recent.close.iloc[-1] - recent.open.iloc[-1]) / spread.iloc[-1])
        if len(recent) and pd.notna(spread.iloc[-1]) else 0.0
    )
    last_upper = float(upper_wick_ratio.iloc[-1]) if len(upper_wick_ratio) else 0.0
    last_lower = float(lower_wick_ratio.iloc[-1]) if len(lower_wick_ratio) else 0.0
    buyer_strength = min(100, max(0, int(max(last_signed_body, 0.0) * 70 + last_lower * 30)))
    seller_strength = min(100, max(0, int(max(-last_signed_body, 0.0) * 70 + last_upper * 30)))
    if buyer_strength - seller_strength >= 15:
        candle_strength_direction, candle_strength_label = "CALL", "FORCA_COMPRADORA"
    elif seller_strength - buyer_strength >= 15:
        candle_strength_direction, candle_strength_label = "PUT", "FORCA_VENDEDORA"
    elif last_lower - last_upper >= 0.20:
        candle_strength_direction, candle_strength_label = "CALL", "REJEICAO_COMPRADORA"
    elif last_upper - last_lower >= 0.20:
        candle_strength_direction, candle_strength_label = "PUT", "REJEICAO_VENDEDORA"
    else:
        candle_strength_direction, candle_strength_label = "NEUTRO", "EQUILIBRIO"

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
        "candle_strength_direction": candle_strength_direction,
        "candle_strength_label": candle_strength_label,
        "buyer_strength_score": buyer_strength,
        "seller_strength_score": seller_strength,
        "price_impact_estimate": round(impact, 3),
        "displacement_atr": round(displacement_atr, 3),
    }


def _flow_components(df: pd.DataFrame, window: int = 12) -> dict:
    """Quatro proxies direcionais OHLCV, sem duplicar votos quando falta volume."""
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


Direction = Literal["CALL", "PUT", "NEUTRO"]


def analyze_candles(candles: list[dict]) -> dict:
    """Analisa velas fechadas mantendo assinatura e contrato do servidor."""
    # EMA 100 só é considerada válida com pelo menos 100 candles válidos.
    if not isinstance(candles, (list, tuple)) or len(candles) < 100:
        return {"ready": False, "reason": "Candles insuficientes para EMA 100"}

    df = pd.DataFrame(candles).copy()
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
        return {"ready": False, "reason": "Candles válidos insuficientes para EMA 100"}

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

    # Direção integrada: tendência independente + proxies disponíveis de fluxo.
    flow_data = _flow_components(df)
    trend_data = _trend_component(close, atr_value)
    flow_direction = flow_data["direction"]
    trend_direction = trend_data["direction"]
    flow = flow_data["pressure"]
    flow_strength = flow_data["strength"]

    # Cada proxy de fluxo entra uma única vez na votação.
    direction_votes = [trend_direction] + flow_data["directional_votes"]
    direction_bullish_votes = direction_votes.count("CALL")
    direction_bearish_votes = direction_votes.count("PUT")
    direction_majority = len(direction_votes) // 2 + 1

    direction: Direction = (
        "CALL" if direction_bullish_votes >= direction_majority else
        "PUT" if direction_bearish_votes >= direction_majority else
        "NEUTRO"
    )
    direction_strength = min(
        100,
        int(
            abs(
                sum(
                    1 if v == "CALL" else -1 if v == "PUT" else 0
                    for v in direction_votes
                ) / len(direction_votes)
            ) * 100
        ),
    )

    # Valores anteriores das EMAs: usados no spread/inclinacao.
    ma9_previous = float(ma9.iloc[i - 3])
    ma21_previous = float(ma21.iloc[i - 3])

    # EMA 100 define o regime direcional e será obrigatória no filtro de entrada.
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

    # EMA 9 + EMA 21: confirmacao conjunta, contando como um unico voto.
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

    k_now, d_now, j_now = float(k.iloc[i]), float(d.iloc[i]), float(j.iloc[i])
    kdj_bull = k_now > d_now and j_now > float(j.iloc[ip])
    kdj_bear = k_now < d_now and j_now < float(j.iloc[ip])
    kdj_direction = "CALL" if kdj_bull else ("PUT" if kdj_bear else "NEUTRO")

    confirms_for = {
        "CALL": [
            macd_bull, rsi_bull, di_bull, adx_confirms,
            kdj_bull, ma_bull, ema100_bull,
        ],
        "PUT": [
            macd_bear, rsi_bear, di_bear, adx_confirms,
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

    pullback_data = _pullback_analysis(
        df=df,
        trend_direction=trend_direction,
        ema21_value=ma21_value,
        atr_value=atr_value,
        fib_zone=fib_zone,
        bullish_reversal=bullish_reversal_candle,
        bearish_reversal=bearish_reversal_candle,
    )
    if pullback_data["detected"] and direction == pullback_data["direction"]:
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

    signal_mode = "FLOW"
    if mean_reversion_direction != "NEUTRO":
        # Reversão pode contrariar a tendência principal, mas não um fluxo
        # estimado fortemente oposto. Fluxo OHLCV fraco/neutro não veta o setup.
        reversal_sign = 1.0 if mean_reversion_direction == "CALL" else -1.0
        strong_opposing_flow = flow * reversal_sign <= -0.35
        if (
            direction == "NEUTRO"
            or direction == mean_reversion_direction
            or not strong_opposing_flow
        ):
            direction = mean_reversion_direction
            signal_mode = "MEAN_REVERSION"
            confirm_count = mean_reversion_confirmations

    # Indicadores confirmam, mas não podem trocar CALL por PUT ou vice-versa.
    if direction != "NEUTRO" and confirm_count < 2:
        direction = "NEUTRO"

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
    market_structure = _market_structure_analysis(df, atr_value)
    brain_score = 0
    if direction != "NEUTRO":
        brain_score = (
            35
            + min(20, flow_strength // 3)
            + min(25, confirm_count * 5)
        )
        brain_score += (
            10 if adx_value >= 25 else
            5 if adx_value >= 20 else
            0
        )
        brain_score += structure_bonus
        if market_structure["direction"] == direction:
            brain_score += 5
        elif market_structure["direction"] in ("CALL", "PUT"):
            brain_score -= 5
        if pullback_data["detected"] and pullback_data["direction"] == direction:
            brain_score += 4
        brain_score -= 15 if absorption_conflict else 0
        brain_score -= 10 if diagnostics["absorption_detected"] else 0
        if signal_mode == "MEAN_REVERSION":
            brain_score += 3
        brain_score = max(0, min(97, brain_score))

        if absorption_conflict and brain_score < 65:
            direction = "NEUTRO"

    # Filtro de qualidade: contexto e tendência/EMA100 pesam na nota,
    # mas deixam de ser vetos absolutos. Só dados/condições realmente ruins
    # bloqueiam a entrada. Mantém a assinatura e os campos esperados pelo servidor.
    entry_filter_reasons = []
    entry_context_warnings = []
    if direction != "NEUTRO":
        if market_structure["direction"] not in ("NEUTRO", direction):
            entry_context_warnings.append("ESTRUTURA_CONTRARIA")
        if market_structure["direction"] == direction:
            entry_context_warnings.append("ESTRUTURA_ALINHADA")
        if pullback_data["detected"] and pullback_data["direction"] == direction:
            entry_context_warnings.append("PULLBACK_CONFIRMADO")
    if direction == "NEUTRO":
        entry_filter_reasons.append("DIRECAO_NAO_CONFIRMADA")
    else:
        direction_sign = 1.0 if direction == "CALL" else -1.0

        # Duas confirmações independentes continuam obrigatórias.
        if confirm_count < 2:
            entry_filter_reasons.append("CONFIRMACOES_INSUFICIENTES")

        # Proteções de volatilidade continuam como bloqueios rígidos.
        if volatility_too_low:
            entry_filter_reasons.append("VOLATILIDADE_BAIXA")
        if volatility_spike:
            entry_filter_reasons.append("PICO_DE_VOLATILIDADE")

        # ADX muito baixo continua bloqueado. Faixa lateral moderada exige
        # confirmação/nota maiores, em vez de vetar qualquer oportunidade.
        if adx_value < 12:
            entry_filter_reasons.append("MERCADO_SEM_FORCA_MINIMA")

        if signal_mode == "MEAN_REVERSION":
            # A reversão não precisa estar alinhada à tendência/EMA100.
            if trend_direction != direction:
                entry_context_warnings.append("REVERSAO_CONTRA_TENDENCIA")
            if ema100_direction != direction:
                brain_score -= 5
                entry_context_warnings.append("EMA100_CONTRARIA_A_REVERSAO")
            if flow * direction_sign <= -0.35:
                entry_filter_reasons.append("FLUXO_FORTE_CONTRA_REVERSAO")
        else:
            if trend_direction != direction:
                brain_score -= 8
                entry_context_warnings.append("TENDENCIA_DIVERGENTE_PENALIZADA")
            if ema100_direction != direction:
                brain_score -= 5
                entry_context_warnings.append("EMA100_DIVERGENTE_PENALIZADA")
            # A direção agregada pode ser válida mesmo se a maioria estrita
            # dos proxies de fluxo estiver neutra. Só oposição forte pesa muito.
            aligned_flow_pressure = flow * direction_sign
            if aligned_flow_pressure <= -0.20:
                brain_score -= 10
                entry_context_warnings.append("FLUXO_CONTRARIO_PENALIZADO")
            elif flow_direction != direction:
                brain_score -= 3
                entry_context_warnings.append("FLUXO_SEM_MAIORIA_DIRECIONAL")

        if sideways:
            brain_score -= 8
            entry_context_warnings.append("ADX_EM_FAIXA_LATERAL")

        brain_score = max(0, min(97, brain_score))
        minimum_score = 63 if sideways else 55
        minimum_confirms = 3 if sideways else 2
        if confirm_count < minimum_confirms:
            entry_filter_reasons.append("CONFIRMACOES_INSUFICIENTES_PARA_ADX")
        if brain_score < minimum_score:
            entry_filter_reasons.append("PONTUACAO_ABAIXO_DO_MINIMO")

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
        "trend_direction": trend_direction,
        "trend_slope_atr": round(trend_data["slope_atr"], 4),
        "trend_component_strength": int(trend_data["strength"]),
        "direction_trend_vote": trend_direction,
        "direction_bullish_votes": int(direction_bullish_votes),
        "direction_bearish_votes": int(direction_bearish_votes),
        "direction_vote_count": int(len(direction_votes)),
        "direction_strength": int(direction_strength),
        "trend_strength": trend_strength,
        "market_structure": market_structure["structure"],
        "market_structure_direction": market_structure["direction"],
        "structure_higher_high": market_structure["higher_high"],
        "structure_higher_low": market_structure["higher_low"],
        "structure_lower_high": market_structure["lower_high"],
        "structure_lower_low": market_structure["lower_low"],
        "structure_breakout": market_structure["breakout"],
        "last_swing_high": market_structure["last_swing_high"],
        "last_swing_low": market_structure["last_swing_low"],
        "pullback_detected": bool(pullback_data["detected"]),
        "pullback_direction": pullback_data["direction"],
        "pullback_zone": pullback_data["zone"],
        "pullback_strength": int(pullback_data["strength"]),
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
        "indicators": indicators,
    }
