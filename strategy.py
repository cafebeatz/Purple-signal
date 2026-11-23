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
    """Analisa candles fechados: vela/fluxo, EMA 9/21, tendência EMA 50/100 e confluências."""
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
    ma9, ma21, ma50, ma100 = (
        ema(close, 9), ema(close, 21), ema(close, 50), ema(close, 100)
    )
    macd_line, macd_signal, macd_hist = macd(close)
    rsi14 = rsi(close, 14)
    k, d, j = kdj(high, low, close, 9)
    adx_series, plus_di, minus_di, atr = _adx(high, low, close, 14)

    i, ip = len(df) - 1, len(df) - 2
    price = float(close.iloc[i])
    atr_value = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
    ma9_value = float(ma9.iloc[i])
    ma21_value = float(ma21.iloc[i])
    ma50_value = float(ma50.iloc[i])
    ma100_value = float(ma100.iloc[i])
    adx_value = float(adx_series.iloc[i])
    dp, dm = float(plus_di.iloc[i]), float(minus_di.iloc[i])
    rsi_value = float(rsi14.iloc[i])

    # ETAPA 1 — leitura da última vela FECHADA (a vela anterior à vela em formação).
    # O chamador deve enviar apenas candles fechados; não há leitura de candles futuros.
    previous_candle = df.iloc[i]
    previous_open = float(previous_candle.open)
    previous_high = float(previous_candle.high)
    previous_low = float(previous_candle.low)
    previous_close = float(previous_candle.close)
    previous_range = max(previous_high - previous_low, 1e-12)
    previous_body = abs(previous_close - previous_open)
    previous_upper_wick = previous_high - max(previous_open, previous_close)
    previous_lower_wick = min(previous_open, previous_close) - previous_low
    previous_candle_direction = (
        "CALL" if previous_close > previous_open else
        "PUT" if previous_close < previous_open else "NEUTRO"
    )
    previous_body_ratio = float(np.clip(previous_body / previous_range, 0.0, 1.0))
    previous_upper_wick_ratio = float(np.clip(previous_upper_wick / previous_range, 0.0, 1.0))
    previous_lower_wick_ratio = float(np.clip(previous_lower_wick / previous_range, 0.0, 1.0))
    previous_wick_pressure = previous_lower_wick_ratio - previous_upper_wick_ratio
    previous_candle_pressure = float(np.clip(
        (1 if previous_candle_direction == "CALL" else
         -1 if previous_candle_direction == "PUT" else 0) * previous_body_ratio * 0.70
        + previous_wick_pressure * 0.30, -1.0, 1.0
    ))

    # ETAPA 2 — fluxo estimado de mercado por corpo, sombras, deslocamento e volume.
    # É uma aproximação OHLCV, não agressão real de compra/venda nem livro de ofertas.
