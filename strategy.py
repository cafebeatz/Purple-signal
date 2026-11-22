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


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """ADX clássico suavizado por Wilder, acompanhado de DI+ e DI-."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_sm = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_sm = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = (100 * plus_sm / atr.replace(0, np.nan)).fillna(0)
    minus_di = (100 * minus_sm / atr.replace(0, np.nan)).fillna(0)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0)
    return adx, plus_di, minus_di, atr


def _flow_diagnostics(df: pd.DataFrame, atr_value: float, window: int = 20) -> dict:
    """Diagnósticos estimados por OHLCV; não substituem agressão nem book real."""
    recent = df.tail(window).copy()
    spread = (recent.high - recent.low).replace(0, np.nan)
    body_ratio = ((recent.close - recent.open).abs() / spread).clip(0, 1).fillna(0)
    volume = pd.to_numeric(
        recent.get("volume", pd.Series(0, index=recent.index)),
        errors="coerce",
    ).fillna(0).clip(lower=0)
    vol_ref = float(volume.median()) if len(volume) else 0.0
    last_vol = float(volume.iloc[-1]) if len(volume) else 0.0
    last_body = float(body_ratio.iloc[-1]) if len(body_ratio) else 0.0
    volume_ratio = last_vol / vol_ref if vol_ref > 0 else 0.0
    absorption = bool(vol_ref > 0 and volume_ratio >= 1.5 and last_body <= 0.30)
    last_range = float(spread.iloc[-1]) if len(spread) and pd.notna(spread.iloc[-1]) else 0.0
    displacement_atr = last_range / atr_value if atr_value > 0 else 0.0
    impact = (last_range / atr_value) / max(volume_ratio, 0.25) if atr_value > 0 else 0.0
    return {
        "absorption_detected": absorption,
        "absorption_strength": round(min(100.0, volume_ratio * (1 - last_body) * 35), 1),
        "volume_ratio": round(volume_ratio, 2),
        "candle_body_ratio": round(last_body, 3),
        "price_impact_estimate": round(impact, 3),
        "displacement_atr": round(displacement_atr, 3),
    }


def _flow_components(df: pd.DataFrame, window: int = 12) -> dict:
    """Quatro proxies direcionais separados, calculados apenas com OHLCV."""
    recent = df.tail(window).copy()
    spread = (recent.high - recent.low).replace(0, np.nan)
    body_pressure = ((recent.close - recent.open) / spread).clip(-1, 1).fillna(0.0)

    volume = pd.to_numeric(
        recent.get("volume", pd.Series(0.0, index=recent.index)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    # 1) Fluxo dos candles: pressão média dos corpos, sem ponderar por volume.
    candle_flow = float(body_pressure.mean()) if len(body_pressure) else 0.0

    # 2) Volume dos candles: pressão dos corpos ponderada pelo volume negociado.
    if float(volume.sum()) > 0:
        candle_volume_flow = float((body_pressure * volume).sum() / volume.sum())
    else:
        candle_volume_flow = candle_flow

    # 3) Fluxo de mercado: deslocamento líquido do preço no período, normalizado
    # pelo caminho percorrido. Evita que uma única vela defina todo o movimento.
    close_change = recent.close.diff().fillna(0.0)
    path = float(close_change.abs().sum())
    market_flow = float(close_change.sum() / path) if path > 0 else 0.0

    # 4) Volume de mercado: desequilíbrio estimado entre volume em barras de alta
    # e de baixa. Sem dados de agressão, é apenas um proxy de volume direcional.
    bar_direction = np.sign(close_change)
    signed_volume = float((bar_direction * volume).sum())
    market_volume_flow = signed_volume / float(volume.sum()) if float(volume.sum()) > 0 else market_flow

    values = {
        "market_flow": float(np.clip(market_flow, -1.0, 1.0)),
        "candle_flow": float(np.clip(candle_flow, -1.0, 1.0)),
        "candle_volume_flow": float(np.clip(candle_volume_flow, -1.0, 1.0)),
        "market_volume_flow": float(np.clip(market_volume_flow, -1.0, 1.0)),
    }
    votes = [1 if value >= 0.08 else (-1 if value <= -0.08 else 0) for value in values.values()]
    bullish = sum(v == 1 for v in votes)
    bearish = sum(v == -1 for v in votes)
    # Exige maioria clara entre as quatro leituras para estabelecer direção.
    direction = "CALL" if bullish >= 3 else ("PUT" if bearish >= 3 else "NEUTRO")
    composite = float(np.mean(list(values.values())))
    strength = min(100, int(abs(composite) * 100))
    return {
        **values,
        "direction": direction,
        "pressure": composite,
        "strength": strength,
        "bullish_votes": bullish,
        "bearish_votes": bearish,
    }


Direction = Literal["CALL", "PUT", "NEUTRO"]


def analyze_candles(candles: list[dict]) -> dict:
    """Analisa velas fechadas mantendo assinatura e contrato do servidor."""
    if not isinstance(candles, (list, tuple)) or len(candles) < 50:
        return {"ready": False, "reason": "Candles insuficientes"}

    df = pd.DataFrame(candles).copy()
    for col in ("open", "high", "low", "close"):
        if col not in df:
            return {"ready": False, "reason": f"Campo obrigatório ausente: {col}"}
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" not in df:
        df["volume"] = 0.0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    valid = (
        (df.high >= df[["open", "close", "low"]].max(axis=1))
        & (df.low <= df[["open", "close", "high"]].min(axis=1))
    )
    df = df.loc[valid].reset_index(drop=True)
    if len(df) < 50:
        return {"ready": False, "reason": "Candles válidos insuficientes"}

    close, high, low = df.close, df.high, df.low
    ma9, ma21 = ema(close, 9), ema(close, 21)
    macd_line, macd_signal, macd_hist = macd(close)
    rsi14 = rsi(close, 14)
    k, d, j = kdj(high, low, close, 9)
    adx_series, plus_di, minus_di, atr = _adx(high, low, close, 14)
    i, ip = len(df) - 1, len(df) - 2

    price = float(close.iloc[i])
    atr_value = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
    ma9_value, ma21_value = float(ma9.iloc[i]), float(ma21.iloc[i])
    adx_value = float(adx_series.iloc[i])
    dp, dm = float(plus_di.iloc[i]), float(minus_di.iloc[i])
    rsi_value = float(rsi14.iloc[i])

    # NÚCLEO 1: quatro leituras independentes estimadas a partir de OHLCV.
    flow_data = _flow_components(df)
    direction: Direction = flow_data["direction"]
    flow_direction = direction
    flow = flow_data["pressure"]
    flow_strength = flow_data["strength"]

    # EMAs não determinam direção. São somente uma confirmação adicional.
    ma_bull = ma9_value > ma21_value and ma9_value > float(ma9.iloc[i - 3])
    ma_bear = ma9_value < ma21_value and ma9_value < float(ma9.iloc[i - 3])
    ema_direction = "CALL" if ma_bull else ("PUT" if ma_bear else "NEUTRO")

    ema_spread_atr = abs(ma9_value - ma21_value) / atr_value if atr_value > 0 else 0.0
    ema_slope_atr = abs(ma9_value - float(ma9.iloc[i - 3])) / atr_value if atr_value > 0 else 0.0
    sideways = bool(adx_value < 16)
    trend_strength = (
        "FORTE" if adx_value >= 25 else
        "MODERADA" if adx_value >= 20 else
        "FRACA" if adx_value >= 16 else
        "LATERAL"
    )

    # Confirmações direcionais. RSI 80/20 identifica extremos; não inverte sinal.
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
        "CALL": [macd_bull, rsi_bull, di_bull, adx_confirms, kdj_bull, ma_bull],
        "PUT": [macd_bear, rsi_bear, di_bear, adx_confirms, kdj_bear, ma_bear],
    }
    confirm_count = sum(bool(x) for x in confirms_for[direction]) if direction != "NEUTRO" else 0

    # NÚCLEO 2: retorno à média pela EMA21. A distância mínima é 1 ATR.
    # Exige candle de reversão e pelo menos duas confirmações independentes.
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
        and (lower_wick / candle_range >= 0.25 or last_close > previous_price)
    )
    bearish_reversal_candle = (
        last_close < last_open
        and (upper_wick / candle_range >= 0.25 or last_close < previous_price)
    )

    mean_reversion_direction = "NEUTRO"
    mean_reversion_confirmations = 0
    if atr_value > 0 and distance_from_ema21_atr <= -1.0 and bullish_reversal_candle:
        mean_reversion_confirmations = sum(
            [macd_bull, rsi_value < 50, di_bull, kdj_bull]
        )
        if mean_reversion_confirmations >= 2:
            mean_reversion_direction = "CALL"
    elif atr_value > 0 and distance_from_ema21_atr >= 1.0 and bearish_reversal_candle:
        mean_reversion_confirmations = sum(
            [macd_bear, rsi_value > 50, di_bear, kdj_bear]
        )
        if mean_reversion_confirmations >= 2:
            mean_reversion_direction = "PUT"

    signal_mode = "FLOW"
    if mean_reversion_direction != "NEUTRO":
        # A reversão só prevalece quando o núcleo de fluxo está neutro ou alinhado.
        if direction == "NEUTRO" or direction == mean_reversion_direction:
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

    brain_score = 0
    if direction != "NEUTRO":
        brain_score = 35 + min(20, flow_strength // 3) + min(25, confirm_count * 5)
        brain_score += 10 if adx_value >= 25 else (5 if adx_value >= 20 else 0)
        brain_score += structure_bonus
        brain_score -= 15 if absorption_conflict else 0
        brain_score -= 10 if diagnostics["absorption_detected"] else 0
        if signal_mode == "MEAN_REVERSION":
            brain_score += 3
        brain_score = max(0, min(97, brain_score))
        if absorption_conflict and brain_score < 65:
            direction = "NEUTRO"

    confidence = int(brain_score) if direction != "NEUTRO" else 0

    macd_cross_up = float(macd_line.iloc[ip]) <= float(macd_signal.iloc[ip]) and macd_bull
    macd_cross_down = float(macd_line.iloc[ip]) >= float(macd_signal.iloc[ip]) and macd_bear
    di_cross_up = float(plus_di.iloc[ip]) <= float(minus_di.iloc[ip]) and di_bull
    di_cross_down = float(plus_di.iloc[ip]) >= float(minus_di.iloc[ip]) and di_bear

    indicators = {
        "flow_direction": flow_direction,
        "flow_pressure": round(flow, 3),
        "flow_strength": flow_strength,
        "flow_source": "ESTIMADO_OHLCV",
        "market_flow": round(flow_data["market_flow"], 3),
        "candle_flow": round(flow_data["candle_flow"], 3),
        "candle_volume_flow": round(flow_data["candle_volume_flow"], 3),
        "market_volume_flow": round(flow_data["market_volume_flow"], 3),
        "flow_bullish_votes": int(flow_data["bullish_votes"]),
        "flow_bearish_votes": int(flow_data["bearish_votes"]),
        "trend_direction": ema_direction,
        "trend_strength": trend_strength,
        "sideways_market": sideways,
        "ema_spread_atr": round(ema_spread_atr, 3),
        "ema_slope_atr": round(ema_slope_atr, 3),
        "ema9": round(ma9_value, 6),
        "ema21": round(ma21_value, 6),
        "ema_confirmation": ema_direction,
        "kdj_k": round(k_now, 2),
        "kdj_d": round(d_now, 2),
        "kdj_j": round(j_now, 2),
        "kdj_direction": kdj_direction,
        "macd": round(float(macd_line.iloc[i]), 6),
        "macd_signal": round(float(macd_signal.iloc[i]), 6),
        "macd_status": (
            "COMPRA" if macd_cross_up else
            "VENDA" if macd_cross_down else
            "ALTA" if macd_bull else "BAIXA"
        ),
        "rsi": round(rsi_value, 2),
        "rsi_status": (
            "SOBRECOMPRA" if rsi_value >= 80 else
            "SOBREVENDA" if rsi_value <= 20 else
            "COMPRA" if rsi_bull else
            "VENDA" if rsi_bear else "NEUTRO"
        ),
        "rsi_period": 14,
        "rsi_overbought": 80,
        "rsi_oversold": 20,
        "adx": round(adx_value, 2),
        "adx_status": "FORTE" if adx_value >= 25 else ("MODERADO" if adx_value >= 20 else "FRACO"),
        "di_plus": round(dp, 2),
        "di_minus": round(dm, 2),
        "didi_status": (
            "AGULHADA COMPRA" if di_cross_up else
            "AGULHADA VENDA" if di_cross_down else
            "DI+ > DI-" if di_bull else "DI- > DI+"
        ),
        "entry_confirmations": int(confirm_count),
        "brain_score": int(brain_score),
        "signal_mode": signal_mode if direction != "NEUTRO" else "NONE",
        "mean_reversion_direction": mean_reversion_direction,
        "mean_reversion_distance_atr": round(distance_from_ema21_atr, 3),
        "mean_reversion_threshold_atr": 1.0,
        "mean_reversion_confirmations": int(mean_reversion_confirmations),
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
