"""Scanner leve e independente do motor operacional Purple Signal.

A pontuação usa somente OHLCV. O payout real é um filtro rígido em conta real;
em modo simulado, payout é desconhecido e o filtro fica explicitamente neutro.
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np
import pandas as pd

MIN_CANDLES = 60
MIN_SCORE = 55
MIN_PAYOUT = 0.86


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def score_asset(
    candles: list[dict],
    payout: float | None = None,
    simulated: bool = False,
    min_score: int = MIN_SCORE,
    min_payout: float = MIN_PAYOUT,
) -> dict:
    """Avalia rapidamente um ativo; não emite sinal CALL/PUT nem autoriza ordens."""
    result = {
        "qualified": False, "score": 0, "direction_bias": "NEUTRO",
        "payout": payout, "payout_ok": bool(simulated or (payout is not None and payout >= min_payout)),
        "criteria": {}, "reason": "DADOS_INSUFICIENTES",
    }
    if len(candles or []) < MIN_CANDLES:
        return result

    df = pd.DataFrame(candles[-60:])
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        result["reason"] = "OHLC_INCOMPLETO"
        return result
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    if len(df) < MIN_CANDLES:
        result["reason"] = "OHLC_INVALIDO"
        return result

    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    ranges = np.maximum(high - low, 1e-12)
    bodies = np.abs(close - open_) / ranges
    signed = np.sign(close - open_)
    nonflat = signed[signed != 0]
    flow = float(np.mean(nonflat[-12:])) if len(nonflat) else 0.0
    direction = "CALL" if flow >= 0.25 else "PUT" if flow <= -0.25 else "NEUTRO"

    # Tendência simples por inclinação da regressão normalizada.
    x = np.arange(len(close), dtype=float)
    slope = float(np.polyfit(x, close, 1)[0])
    price_scale = max(abs(float(np.mean(close))), 1e-12)
    trend_strength = abs(slope) * (len(close) - 1) / price_scale
    trend_direction = "CALL" if slope > 0 else "PUT" if slope < 0 else "NEUTRO"

    criteria: dict[str, dict] = {}
    score = 0.0

    flow_points = min(20.0, abs(flow) * 25.0)
    criteria["fluxo_candles"] = {"value": round(flow, 3), "points": round(flow_points, 1), "max": 20}
    score += flow_points

    trend_points = min(20.0, trend_strength * 1800.0)
    criteria["tendencia"] = {
        "direction": trend_direction, "strength": round(trend_strength, 5),
        "points": round(trend_points, 1), "max": 20,
    }
    score += trend_points

    volume = pd.to_numeric(df.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).clip(lower=0)
    if len(volume) and float(volume.median()) > 0:
        vol_ratio = float(volume.tail(5).mean() / max(float(volume.median()), 1e-12))
        volume_points = 15.0 if vol_ratio >= 0.8 else 10.0 if vol_ratio >= 0.5 else 3.0
    else:
        vol_ratio, volume_points = None, 7.5  # feed sem volume: neutro, não reprova sozinho
    criteria["volume"] = {"ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
                          "points": volume_points, "max": 15}
    score += volume_points

    returns = np.diff(close) / np.maximum(np.abs(close[:-1]), 1e-12)
    volatility = float(np.std(returns[-20:])) if len(returns) >= 20 else 0.0
    volatility_points = 15.0 if volatility > 0.00002 else 7.0 if volatility > 0.000005 else 0.0
    criteria["volatilidade"] = {"return_std": round(volatility, 7),
                                "points": volatility_points, "max": 15}
    score += volatility_points

    body_mean = float(np.mean(bodies[-12:]))
    strength_points = min(15.0, body_mean * 18.75)
    criteria["forca_candles"] = {"body_ratio": round(body_mean, 3),
                                 "points": round(strength_points, 1), "max": 15}
    score += strength_points

    # Regularidade: penaliza alternância excessiva e pavios dominantes.
    last_signs = np.sign(np.diff(close[-13:]))
    alternation = float(np.mean(last_signs[1:] != last_signs[:-1])) if len(last_signs) > 1 else 0.5
    wick_ratio = float(np.mean(1.0 - bodies[-12:]))
    regularity = max(0.0, 1.0 - 0.65 * alternation - 0.35 * wick_ratio)
    regularity_points = regularity * 15.0
    criteria["regularidade"] = {"alternation": round(alternation, 3),
                                "wick_ratio": round(wick_ratio, 3),
                                "points": round(regularity_points, 1), "max": 15}
    score += regularity_points

    payout_ok = simulated or (payout is not None and payout >= min_payout)
    qualified = payout_ok and score >= min_score
    result.update({
        "qualified": bool(qualified), "score": round(score, 1),
        "direction_bias": direction if direction != "NEUTRO" else trend_direction,
        "payout": payout, "payout_ok": bool(payout_ok),
        "criteria": criteria,
        "reason": "APROVADO_NO_SCANNER" if qualified else
                  "PAYOUT_ABAIXO_DE_86" if not payout_ok and payout is not None else
                  "PAYOUT_NAO_DISPONIVEL" if not payout_ok else "PONTUACAO_INSUFICIENTE",
    })
    return result
